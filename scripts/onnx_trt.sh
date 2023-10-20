#!/bin/bash
# Run the ONNX model with trtexec
# precisions: fp32, fp16, int8, best

# Check if the input argument is provided
if [ -z "$1" ]
    then
        echo "Please provide the path to the ONNX model as the first argument"
        exit 1
fi
if [ ! -f "$1" ]
    then
        echo "File $1 does not exist"
        exit 1
fi

# Create the directories for the results
mkdir -p result/fp32
mkdir -p result/fp16
mkdir -p result/int8
mkdir -p result/best
mkdir -p temp/fp32
mkdir -p temp/fp16
mkdir -p temp/int8
mkdir -p temp/best

# fp32
# CUDA
trtexec --onnx="$1" --saveEngine="temp/fp32/cuda.trt" \
    --exportTimes="result/fp32/times_cuda.json" \
    --exportOutput="result/fp32/output_cuda.json" \
    --exportProfile="result/fp32/profile_cuda.json" \
    --exportLayerInfo="result/fp32/layerinfo_cuda.json" \
    > result/fp32/dump_cuda.txt
# DLA
trtexec --onnx="$1" --saveEngine="temp/fp32/dla.trt" \
    --allowGPUFallback --useDLACore=0 \
    --exportTimes="result/fp32/times_dla.json" \
    --exportOutput="result/fp32/output_dla.json" \
    --exportProfile="result/fp32/profile_dla.json" \
    --exportLayerInfo="result/fp32/layerinfo_dla.json" \
    > result/fp32/dump_dla.txt

for precision in fp16 int8 best
do
    # CUDA
    trtexec --onnx="$1" --saveEngine="temp/$precision/cuda.trt" --$precision \
    --exportTimes="result/$precision/times_cuda.json" \
    --exportOutput="result/$precision/output_cuda.json" \
    --exportProfile="result/$precision/profile_cuda.json" \
    --exportLayerInfo="result/$precision/layerinfo_cuda.json" \
    > result/$precision/dump_cuda.txt
    # DLA
    trtexec --onnx="$1" --saveEngine="temp/$precision/dla.trt" --$precision \
    --allowGPUFallback --useDLACore=0 \
    --exportTimes="result/$precision/times_dla.json" \
    --exportOutput="result/$precision/output_dla.json" \
    --exportProfile="result/$precision/profile_dla.json" \
    --exportLayerInfo="result/$precision/layerinfo_dla.json" \
    > result/$precision/dump_dla.txt
done