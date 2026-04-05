#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

tracker_type=siamese
tracker_type=odtrack

if [ $tracker_type == "siamese" ]; then
    model_name="siamrpn_alex_dwxcorr"
    #model_name="siamrpn_alex_dwxcorr_otb"
    #model_name="siamrpn_r50_l234_dwxcorr"
    #model_name="siamrpn_r50_l234_dwxcorr_otb"
    #model_name="siamrpn_r50_l234_dwxcorr_lt"
    #model_name="siamrpn_mobilev2_l234_dwxcorr"
    #model_name="siammask_r50_l3"
    config_file="experiments/${model_name}/config.yaml"
    model_path="datas/models/${model_name}/model.pth"

elif [ $tracker_type == "odtrack" ]; then
    model_name="Base-Fulldata-300ep"
    config_file="experiments/odtrack/${model_name}.yaml"
    model_path="datas/models/odtrack/${model_name}/model.pth.tar"

else
    echo "Unsupported tracker type: ${tracker_type}"
    exit 1
fi

video_path=""
video_path="demo/bag.avi"
#video_path="../pytracking/datas/videos/UavData/uav_20260330.mp4"

python tools/demo.py \
    --tracker_type=${tracker_type} \
    --config_file=${config_file} \
    --model_path=${model_path} \
    --video_path=${video_path}

