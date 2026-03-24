#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

model_name=siamrpn_alex_dwxcorr

python tools/export.py \
    --config experiments/${model_name}/config.yaml \
    --snapshot datas/models/${model_name}/model.pth \
    --output datas/models/${model_name}/model_scripted.pt


