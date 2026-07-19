#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Configuration
#tracker_type=siamese
tracker_type=odtrack

# TensorRT options
use_trtexec=true       # Use trtexec command line tool (true/false)
enable_fp16=true       # Enable FP16 precision for faster inference
enable_int8=false      # Enable INT8 precision (requires calibration)
max_batch_size=1       # Maximum batch size
workspace_size=4096    # Workspace size in MB

if [ $tracker_type == "siamese" ]; then
    model_name="siamrpn_alex_dwxcorr"
    #model_name="siamrpn_alex_dwxcorr_otb"
    #model_name="siamrpn_r50_l234_dwxcorr"
    #model_name="siamrpn_r50_l234_dwxcorr_otb"
    #model_name="siamrpn_mobilev2_l234_dwxcorr"
    #model_name="siammask_r50_l3"
    snapshot="datas/models/${model_name}/model.pth"
    output="datas/models/${model_name}/model.engine"

elif [ $tracker_type == "odtrack" ]; then
    model_name="odtrack_base_fulldata_300ep"
    snapshot="datas/models/odtrack/Base-Fulldata-300ep/model.pth.tar"
    output="datas/models/odtrack/Base-Fulldata-300ep/model.engine"
    #model_name="odtrack_large_fulldata_300ep"
    #snapshot="datas/models/odtrack/Large-Fulldata-300ep/model.pth.tar"
    #output="datas/models/odtrack/Large-Fulldata-300ep/model.engine"

else
    echo "Unsupported tracker type: ${tracker_type}"
    exit 1
fi

# Build command
cmd="python tools/export_trt.py \
    --config ${model_name} \
    --snapshot ${snapshot} \
    --output ${output} \
    --max-batch-size ${max_batch_size} \
    --workspace-size ${workspace_size}"

# Add precision flags
if [ "$enable_fp16" = true ]; then
    cmd="$cmd --fp16"
fi

if [ "$enable_int8" = true ]; then
    cmd="$cmd --int8"
fi

# Add trtexec flag
if [ "$use_trtexec" = true ]; then
    cmd="$cmd --use-trtexec"
fi

echo "========================================"
echo "TensorRT Export Configuration"
echo "========================================"
echo "Tracker Type:    ${tracker_type}"
echo "Model Name:      ${model_name}"
echo "Snapshot:        ${snapshot}"
echo "Output:          ${output}"
echo "Method:          $([ "$use_trtexec" = true ] && echo "trtexec" || echo "Python API")"
echo "Precision:       $([ "$enable_fp16" = true ] && echo "FP16" || ([ "$enable_int8" = true ] && echo "INT8" || echo "FP32"))"
echo "Max Batch Size:  ${max_batch_size}"
echo "Workspace Size:  ${workspace_size} MB"
echo "========================================"
echo ""

# Execute
eval $cmd
