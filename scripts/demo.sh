#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_name=siamrpn_alex_dwxcorr
#model_name=siamrpn_alex_dwxcorr_otb
#model_name="siamrpn_r50_l234_dwxcorr"
#model_name="siamrpn_r50_l234_dwxcorr_otb"
#model_name="siamrpn_r50_l234_dwxcorr_lt"
#model_name="siamrpn_mobilev2_l234_dwxcorr"
#model_name="siammask_r50_l3"

video_path=demo/bag.avi
#video_path="../pytracking/datas/videos/UavData/uav_20260330.mp4"

python tools/demo.py \
    --config experiments/${model_name}/config.yaml \
    --snapshot datas/models/${model_name}/model.pth \
    --video ${video_path}

