#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

#tracker_type=siamese
tracker_type=odtrack

if [ $tracker_type == "siamese" ]; then
    model_name="siamrpn_alex_dwxcorr"
    #model_name="siamrpn_alex_dwxcorr_otb"
    #model_name="siamrpn_r50_l234_dwxcorr"
    #model_name="siamrpn_r50_l234_dwxcorr_otb"
    #model_name="siamrpn_r50_l234_dwxcorr_lt"
    #model_name="siamrpn_mobilev2_l234_dwxcorr"
    #model_name="siammask_r50_l3"
    snapshot="datas/models/${model_name}/model.pth"
    output="datas/models/${model_name}/model.pt"

elif [ $tracker_type == "odtrack" ]; then
    model_name="odtrack_base_fulldata_300ep"
    snapshot="datas/models/odtrack/Base-Fulldata-300ep/model.pth.tar"
    output="datas/models/odtrack/Base-Fulldata-300ep/model.pt"
    #model_name="odtrack_large_fulldata_300ep"
    #snapshot="datas/models/odtrack/Large-Fulldata-300ep/model.pth.tar"
    #output="datas/models/odtrack/Large-Fulldata-300ep/model.pt"

else
    echo "Unsupported tracker type: ${tracker_type}"
    exit 1
fi

python tools/export.py \
    --config ${model_name} \
    --snapshot ${snapshot} \
    --output ${output}


