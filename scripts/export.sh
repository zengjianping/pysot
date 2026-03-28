#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_name="siamrpn_alex_dwxcorr"
#model_name="siamrpn_alex_dwxcorr_otb"
#model_name="siamrpn_r50_l234_dwxcorr"
#model_name="siamrpn_r50_l234_dwxcorr_otb"
#model_name="siamrpn_r50_l234_dwxcorr_lt"
#model_name="siamrpn_mobilev2_l234_dwxcorr"
#model_name="siammask_r50_l3"

python tools/export.py \
    --config ${model_name} \
    --snapshot datas/models/${model_name}/model.pth \
    --output datas/models/${model_name}/model.pt


