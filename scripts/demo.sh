#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_name=siamrpn_alex_dwxcorr
#model_name=siamrpn_alex_dwxcorr_otb
#model_name="siamrpn_r50_l234_dwxcorr"
#model_name="siamrpn_r50_l234_dwxcorr_otb"
#model_name="siamrpn_mobilev2_l234_dwxcorr"

python tools/demo.py \
    --config experiments/${model_name}/config.yaml \
    --snapshot datas/models/${model_name}/model.pth \
    --video demo/bag.avi

