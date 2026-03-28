import os, sys, cv2
import argparse
import torch
import numpy as np
from glob import glob
from typing import Tuple, List, Optional

import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck


class TrackerModel(nn.Module):
    def __init__(self, 
            backbone_type=None, backbone_kwargs=None,
            rpn_type=None, rpn_kwargs=None,
            neck_type=None, neck_kwargs=None,
            mask_type=None, mask_kwargs=None,
            refine_type=None, refine_kwargs=None):
        super(TrackerModel, self).__init__()
        
        self.neck = None
        self.mask_head = None
        self.refine_head = None

        # build backbone
        self.backbone = get_backbone(backbone_type, **backbone_kwargs)

        # build adjust layer
        if neck_type is not None:
            self.neck = get_neck(neck_type, **neck_kwargs)

        # build rpn head
        self.rpn_head = get_rpn_head(rpn_type, **rpn_kwargs)

        # build mask head
        if mask_type is not None:
            self.mask_head = get_mask_head(mask_type, **mask_kwargs)
            if refine_type is not None:
                self.refine_head = get_refine_head(refine_type, **refine_kwargs)

    def _template(self, z):
        zf = self.backbone(z)
        if self.mask_head is not None:
            zf = zf[-1]
        if self.neck is not None:
            zf = self.neck(zf)
        self.zf = zf

    def _track(self, x):
        xf = self.backbone(x)
        if self.mask_head is not None:
            self.xf = xf[:-1]
            xf = xf[-1]
        if self.neck is not None:
            xf = self.neck(xf)
        cls, loc = self.rpn_head(self.zf, xf)
        if self.mask_head is not None:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
            return (cls, loc, mask)
        else:
            return (cls, loc)

    def _mask_refine(self, pos: List[int]) -> torch.Tensor:
        return self.refine_head(self.xf, self.mask_corr_feature, pos)


class DaSiamRPNModel(TrackerModel):
    def __init__(self, model_params:dict):
        super(DaSiamRPNModel, self).__init__(**model_params)

    @torch.jit.export
    def extract_template(self, z: torch.Tensor) -> None:
        self._template(z)

    @torch.jit.export
    def track(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._track(x)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extract_template(z)
        return self.track(x)


class SiamRPNppModel(TrackerModel):
    def __init__(self, model_params:dict):
        super(SiamRPNppModel, self).__init__(**model_params)

    @torch.jit.export
    def extract_template(self, z: torch.Tensor) -> None:
        self._template(z)

    @torch.jit.export
    def track(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._track(x)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.extract_template(z)
        return self.track(x)


class SiamMaskModel(TrackerModel):
    def __init__(self, model_params:dict):
        super(SiamMaskModel, self).__init__(**model_params)

    @torch.jit.export
    def extract_template(self, z: torch.Tensor) -> None:
        self._template(z)

    @torch.jit.export
    def track(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._track(x)

    @torch.jit.export
    def mask_refine(self, pos: List[int]) -> torch.Tensor:
        return self._mask_refine(pos)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.extract_template(z)
        return self.track(x)


MODELS = {
    "siamrpn_alex_dwxcorr": {
        "model_type": DaSiamRPNModel,
        "model_params": {
            'backbone_type': 'alexnetlegacy',
            'backbone_kwargs': {'width_mult': 1.0},
            'rpn_type': 'DepthwiseRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': 256, 'out_channels': 256}
        }
    },
    "siamrpn_alex_dwxcorr_otb": {
        "model_type": DaSiamRPNModel,
        "model_params": {
            'backbone_type': 'alexnetlegacy',
            'backbone_kwargs': {'width_mult': 1.0},
            'rpn_type': 'DepthwiseRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': 256, 'out_channels': 256}
        }
    },
    "siamrpn_r50_l234_dwxcorr": {
        "model_type": SiamRPNppModel,
        "model_params": {
            'backbone_type': 'resnet50',
            'backbone_kwargs': {'used_layers': [2, 3, 4]},
            'rpn_type': 'MultiRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': True},
            'neck_type': 'AdjustAllLayer',
            'neck_kwargs': {'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]}
        }
    },
    "siamrpn_r50_l234_dwxcorr_otb": {
        "model_type": SiamRPNppModel,
        "model_params": {
            'backbone_type': 'resnet50',
            'backbone_kwargs': {'used_layers': [2, 3, 4]},
            'rpn_type': 'MultiRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': False},
            'neck_type': 'AdjustAllLayer',
            'neck_kwargs': {'in_channels': [512, 1024, 2048], 'out_channels': [256, 256, 256]}
        }
    },
    "siamrpn_r50_l234_dwxcorr_lt": {
        "model_type": SiamRPNppModel,
        "model_params": {
            'backbone_type': 'resnet50',
            'backbone_kwargs': {'used_layers': [2, 3, 4]},
            'rpn_type': 'MultiRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': [128, 256, 512], 'weighted': True},
            'neck_type': 'AdjustAllLayer',
            'neck_kwargs': {'in_channels': [512, 1024, 2048], 'out_channels': [128, 256, 512]}
        }
    },
    "siamrpn_mobilev2_l234_dwxcorr": {
        "model_type": SiamRPNppModel,
        "model_params": {
            'backbone_type': 'mobilenetv2',
            'backbone_kwargs': {'used_layers': [3, 5, 7], 'width_mult': 1.4},
            'rpn_type': 'MultiRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': [256, 256, 256], 'weighted': False},
            'neck_type': 'AdjustAllLayer',
            'neck_kwargs': {'in_channels': [44, 134, 448], 'out_channels': [256, 256, 256]}
        }
    },
    "siammask_r50_l3": {
        "model_type": SiamMaskModel,
        "model_params": {
            'backbone_type': 'resnet50',
            'backbone_kwargs': {'used_layers': [0, 1, 2, 3]},
            'rpn_type': 'DepthwiseRPN',
            'rpn_kwargs': {'anchor_num': 5, 'in_channels': 256, 'out_channels': 256},
            'neck_type': 'AdjustAllLayer',
            'neck_kwargs': {'in_channels': [1024], 'out_channels': [256]},
            'mask_type': 'MaskCorr',
            'mask_kwargs': {'in_channels': 256, 'hidden': 256, 'out_channels': 3969},
            'refine_type': 'Refine',
            'refine_kwargs': {}
        }
    }
}

def export(model_name: str, weights_path: str, output_path: str, device: str = "cpu") -> None:
    # create model
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS)}")
    model_params = MODELS[model_name]["model_params"]
    model = MODELS[model_name]["model_type"](model_params)
    print(f"Model architecture:\n{model}")

    # load model
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        print("Original state dict keys:")
        for k in state.keys():
            print(f"  {k}")
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f"Loaded weights from {weights_path}")
    else:
        print("No weights provided, exporting with random initialization")

    model = model.to(device).eval()

    z = torch.zeros(1, 3, 127, 127, device=device)
    x = torch.zeros(1, 3, 255, 255, device=device)
    with torch.no_grad():
        model.extract_template(z)
        out = model.track(x)
        if isinstance(out, tuple):
            print(f"cls shape: {out[0].shape}, loc shape: {out[1].shape}")
        else:
            print(f"score map shape: {out.shape}")

    #scripted = torch.jit.trace(model, example_input)
    scripted = torch.jit.script(model)
    scripted.save(output_path)
    print(f"Saved TorchScript model -> {output_path}")

# 导出scripted model
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--config', type=str, help='model config')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--output', type=str, help='output path')
    args = parser.parse_args()
    
    export(args.config, args.snapshot, args.output)

