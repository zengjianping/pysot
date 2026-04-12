import os, sys
import torch
from .base_tracker import BaseTracker


class TrackerFactory(object):
    """Tracker factory for creating tracker instances."""

    @staticmethod
    def create_instance(track_type: str, config_file: str, model_path: str) -> BaseTracker:
        if track_type == 'siamese':
            from ..core.config import cfg
            from ..models.model_builder import ModelBuilder
            from .tracker_builder import build_tracker

            # load config
            cfg.merge_from_file(config_file)
            cfg.CUDA = torch.cuda.is_available() and cfg.CUDA
            device = torch.device('cuda' if cfg.CUDA else 'cpu')

            # create model
            model = ModelBuilder()

            # load model
            model.load_state_dict(torch.load(model_path,
                map_location=lambda storage, loc: storage.cpu()))
            model.eval().to(device)

            # build tracker
            tracker = build_tracker(model)
            return tracker

        elif track_type == 'odtrack':
            from ..odtrack.tracker import build_tracker
            tracker = build_tracker(config_file, model_path)
            return tracker

        elif track_type == 'odtracker':
            from ..odtrack.tracker import ODTracker
            tracker = ODTracker(config_file, model_path)
            return tracker

        else:
            raise NotImplementedError
