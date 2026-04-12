from .odtracker import ODTracker
from .odtrack import ODTrack
from .parameter import get_parameters

def build_tracker(config_file: str, model_file: str):
    params = get_parameters(config_file, model_file)
    params.tracker_name = "odtrack"
    params.param_name = config_file
    tracker = ODTrack(params)
    return tracker

