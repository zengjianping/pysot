from .odtracker import ODTracker
from .odtrack import ODTrack
from .parameter import get_parameters

def build_tracker(config_file: str, model_file: str, mode: int = 0):
    params = get_parameters(config_file, model_file)
    params.tracker_name = "odtrack"
    params.param_name = config_file
    if mode == 0:
        tracker = ODTrack(params)
    else:
        tracker = ODTracker(params)
    return tracker

