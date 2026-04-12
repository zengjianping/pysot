import os, cv2
import numpy as np
import torch

from ...tracker.base_tracker import BaseTracker
from .data_utils import Preprocessor
from ..models import ODTrackModel
from ..utils.hann import hann2d
from ..utils.processing_utils import sample_target
from ..utils.processing_utils import transform_image_to_crop
from ..utils.box_ops import clip_box
from ..utils.ce_utils import generate_mask_cond


class ODTracker(BaseTracker):
    def __init__(self, params):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        network = ODTrackModel()
        model_state = torch.load(params.checkpoint, map_location='cpu')['net']
        network.load_state_dict(model_state, strict=True)
        self.network = network.to(self.device)
        self.network.eval()

        self.params = params
        self.cfg = params.cfg
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)

    def init(self, image, bbox):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.memory_frames = [template.tensors]

        self.memory_masks = []
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(bbox, resize_factor, template.tensors.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox))
        
        # save states
        self.state = bbox
        self.frame_id = 0

    def track(self, image):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # --------- select memory frames ---------
        box_mask_z = None
        if self.frame_id <= self.cfg.TEST.TEMPLATE_NUMBER:
            template_list = self.memory_frames.copy()
            if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
                box_mask_z = torch.cat(self.memory_masks, dim=1)
        else:
            template_list, box_mask_z = self.select_memory_frames()
        # --------- select memory frames ---------

        with torch.no_grad():
            out_dict = self.network.forward(template=template_list, search=search.tensors, ce_template_mask=box_mask_z)

        if isinstance(out_dict, list):
            out_dict = out_dict[-1]
            
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.cal_bbox(response, out_dict['size_map'],
            out_dict['offset_map'], return_score=True)
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # --------- save memory frames and masks ---------
        z_patch_arr, z_resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        cur_frame = self.preprocessor.process(z_patch_arr, z_amask_arr)
        frame = cur_frame.tensors
        if self.frame_id > self.cfg.TEST.MEMORY_THRESHOLD:
            frame = frame.detach().cpu()
        self.memory_frames.append(frame)
        if self.cfg.MODEL.BACKBONE.CE_LOC:  # use CE module
            template_bbox = self.transform_bbox_to_crop(self.state, z_resize_factor, frame.device).squeeze(1)
            self.memory_masks.append(generate_mask_cond(self.cfg, 1, frame.device, template_bbox))
        if 'pred_iou' in out_dict.keys():      # use IoU Head
            pred_iou = out_dict['pred_iou'].squeeze(-1)
            self.memory_ious.append(pred_iou)
        # --------- save memory frames and masks ---------

        return {"bbox": self.state, 'best_score': best_score}

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        stride = self.cfg.MODEL.BACKBONE.STRIDE
        feat_sz = int(self.cfg.DATA.SEARCH.SIZE / stride)
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // feat_sz
        idx_x = idx % feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def transform_bbox_to_crop(self, box_in, resize_factor, device, box_extract=None, crop_type='template'):
        # box_in: list [x1, y1, w, h], not normalized
        # box_extract: same as box_in
        # out bbox: Torch.tensor [1, 1, 4], x1y1wh, normalized
        if crop_type == 'template':
            crop_sz = torch.Tensor([self.params.template_size, self.params.template_size])
        elif crop_type == 'search':
            crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        else:
            raise NotImplementedError

        box_in = torch.tensor(box_in)
        if box_extract is None:
            box_extract = box_in
        else:
            box_extract = torch.tensor(box_extract)
        template_bbox = transform_image_to_crop(box_in, box_extract, resize_factor, crop_sz, normalize=True)
        template_bbox = template_bbox.view(1, 1, 4).to(device)

        return template_bbox

    def select_memory_frames(self):
        num_segments = self.cfg.TEST.TEMPLATE_NUMBER
        cur_frame_idx = self.frame_id
        if num_segments != 1:
            assert cur_frame_idx > num_segments
            dur = cur_frame_idx // num_segments
            indexes = np.concatenate([
                np.array([0]),
                np.array(list(range(num_segments))) * dur + dur // 2
            ])
        else:
            indexes = np.array([0])
        indexes = np.unique(indexes)

        select_frames, select_masks = [], []
        
        for idx in indexes:
            frames = self.memory_frames[idx]
            #if not frames.is_cuda:
            #    frames = frames.cuda()
            frames = frames.to(self.device)
            select_frames.append(frames)
            
            if self.cfg.MODEL.BACKBONE.CE_LOC:
                box_mask_z = self.memory_masks[idx]
                select_masks.append(box_mask_z.to(self.device))
        
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            return select_frames, torch.cat(select_masks, dim=1)
        else:
            return select_frames, None
    
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

