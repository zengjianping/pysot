import os, sys, cv2
import torch
import numpy as np
from easydict import EasyDict as edict
from typing import Tuple, List, Dict, Optional, Union

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_clones

from .layers.head import build_box_head
from .layers.vit import vit_base_patch16_224, vit_large_patch16_224
from .layers.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce


base_model_config = {
    "DATA": {
        "MAX_SAMPLE_INTERVAL": 400,
        "MEAN": [0.485, 0.456, 0.406],
        "STD": [0.229, 0.224, 0.225],
        "SEARCH": {
            "CENTER_JITTER": 4.5,
            "FACTOR": 5.0,
            "SCALE_JITTER": 0.5,
            "SIZE": 384,
            "NUMBER": 2
        },
        "TEMPLATE": {
            "CENTER_JITTER": 0,
            "FACTOR": 2.0,
            "SCALE_JITTER": 0,
            "SIZE": 192,
            "NUMBER": 3
        }
    },
    "MODEL": {
        "PRETRAIN_FILE": "mae_pretrain_vit_base.pth",
        "EXTRA_MERGER": False,
        "RETURN_INTER": False,
        "RETURN_STAGES": [],
        "BACKBONE": {
            "TYPE": "vit_base_patch16_224_ce",
            "STRIDE": 16,
            "CE_LOC": [3, 6, 9],
            "CE_KEEP_RATIO": [0.7, 0.7, 0.7],
            "CE_TEMPLATE_RANGE": "CTR_POINT",  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX
            "ADD_CLS_TOKEN": True,             # use track_query mechanism
            "ATTN_TYPE": "concat",             # Choose from [concat, separate]
            "CAT_MODE": "direct",
            "MID_PE": False,
            "SEP_SEG": False,
            "TOKEN_LEN": 1
        },
        "HEAD": {
            "TYPE": "CENTER",
            "NUM_CHANNELS": 256
        }
    },
    "TRAIN": {
        "BBOX_TASK": True,
        "BACKBONE_MULTIPLIER": 0.1,
        "DROP_PATH_RATE": 0.1
    }
}

large_model_config = {
    "DATA": {
        "MAX_SAMPLE_INTERVAL": 400,
        "MEAN": [0.485, 0.456, 0.406],
        "STD": [0.229, 0.224, 0.225],
        "SEARCH": {
            "CENTER_JITTER": 4.5,
            "FACTOR": 5.0,
            "SCALE_JITTER": 0.5,
            "SIZE": 384,
            "NUMBER": 2
        },
        "TEMPLATE": {
            "CENTER_JITTER": 0,
            "FACTOR": 2.0,
            "SCALE_JITTER": 0,
            "SIZE": 192,
            "NUMBER": 3
        }
    },
    "MODEL": {
        "PRETRAIN_FILE": "mae_pretrain_vit_large.pth",
        "EXTRA_MERGER": False,
        "RETURN_INTER": False,
        "RETURN_STAGES": [],
        "BACKBONE": {
            "TYPE": "vit_large_patch16_224_ce",
            "STRIDE": 16,
            "CE_LOC": [3, 6, 9],
            "CE_KEEP_RATIO": [0.7, 0.7, 0.7],
            "CE_TEMPLATE_RANGE": "CTR_POINT",  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX
            "ADD_CLS_TOKEN": True,             # use track_query mechanism
            "ATTN_TYPE": "concat",             # Choose from [concat, separate]
            "CAT_MODE": "direct",
            "MID_PE": False,
            "SEP_SEG": False,
            "TOKEN_LEN": 1
        },
        "HEAD": {
            "TYPE": "CENTER",
            "NUM_CHANNELS": 256
        }
    },
    "TRAIN": {
        "BBOX_TASK": True,
        "BACKBONE_MULTIPLIER": 0.1,
        "DROP_PATH_RATE": 0.1
    }
}

model_configs = {
    "base": base_model_config,
    "large": large_model_config
}

class ODTrackModel(nn.Module):
    def __init__(self, model_params:dict):
        super(ODTrackModel, self).__init__()

        model_type = model_params['model_type']
        model_config = model_configs[model_type]
        cfg = edict(model_config)
        pretrained = ''

        if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
            backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

        elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
            backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                            )
            
        elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
            backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

        elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
            backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                                )

        else:
            raise NotImplementedError

        hidden_dim = backbone.embed_dim
        patch_start_index = 1
        
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
        box_head = build_box_head(cfg, hidden_dim)
        self.backbone = backbone
        self.box_head = box_head

        head_type = cfg.MODEL.HEAD.TYPE
        token_len = cfg.MODEL.BACKBONE.TOKEN_LEN

        self.aux_loss = False
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    @torch.jit.export
    def forward(self, template: List[torch.Tensor], search: torch.Tensor,
                ce_template_mask: Optional[torch.Tensor] = None, 
                ce_keep_rate: Optional[float] = None, 
                return_last_attn: bool = False,
            ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        x, aux_dict = self.backbone(z=template.copy(), x=search,
            ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
            return_last_attn=return_last_attn, track_query=self.track_query,
            token_len=self.token_len)

        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
            
        enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        if self.backbone.add_cls_token:
            self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)
            
        att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
        opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        
        # Forward head
        """enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)"""
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        # run the center head
        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, None)
        
        # outputs_coord = box_xyxy_to_cxcywh(bbox)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        
        out:Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            'pred_boxes': outputs_coord_new,
            'score_map': score_map_ctr,
            'size_map': size_map,
            'offset_map': offset_map
        }

        out.update(aux_dict)
        out['backbone_feat'] = x
        
        return out

