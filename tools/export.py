import os
import argparse

import cv2
import torch
import numpy as np
from glob import glob

from typing import Tuple, Optional

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

class TrackerModel(ModelBuilder):
    def __init__(self):
        super(TrackerModel, self).__init__()

    @torch.jit.export
    def extract_template(self, z: torch.Tensor) -> None:
        self.template(z)

    @torch.jit.export
    def track_object(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        result = self.track(x)
        cls = result['cls']
        loc = result['loc']
        mask = result['mask']
        return cls, loc, mask

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.extract_template(z)
        return self.track_object(x)

def export(config_file: str, weights_path: str, output_path: str,
           device: str = "cpu") -> None:

    # load config
    cfg.merge_from_file(config_file)

    # create model
    model = TrackerModel()
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
    x = torch.zeros(1, 3, 287, 287, device=device)
    with torch.no_grad():
        model.extract_template(z)
        out = model.track_object(x)
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
    parser.add_argument('--config', type=str, help='config file')
    parser.add_argument('--snapshot', type=str, help='model name')
    parser.add_argument('--output', type=str, help='output path')
    args = parser.parse_args()
    
    export(args.config, args.snapshot, args.output)

