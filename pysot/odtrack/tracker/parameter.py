import os
from ..utils.params import TrackerParams
from .config import cfg, update_config_from_file


def get_parameters(yaml_file: str, checkpoint_path: str):
    params = TrackerParams()
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = checkpoint_path

    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params
