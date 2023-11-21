#!/bin/bash
# Run the ONNX model with trtexec
# precisions: fp32, fp16, int8, best

result_dir="result/$1"
trt_dir="temp/$1"

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
mkdir -p ${result_dir}/fp32
mkdir -p ${result_dir}/fp16
mkdir -p ${result_dir}/int8
mkdir -p ${result_dir}/best
mkdir -p ${trt_dir}/fp32
mkdir -p ${trt_dir}/fp16
mkdir -p ${trt_dir}/int8
mkdir -p ${trt_dir}/best

# fp32
# CUDA
trtexec --onnx="$1" --saveEngine="${trt_dir}/fp32/cuda.trt" \
    --exportTimes="${result_dir}/fp32/times_cuda.json" \
    --exportOutput="${result_dir}/fp32/output_cuda.json" \
    --exportProfile="${result_dir}/fp32/profile_cuda.json" \
    --exportLayerInfo="${result_dir}/fp32/layerinfo_cuda.json" \
    > ${result_dir}/fp32/dump_cuda.txt
# DLA
trtexec --onnx="$1" --saveEngine="${trt_dir}/fp32/dla.trt" \
    --allowGPUFallback --useDLACore=0 \
    --exportTimes="${result_dir}/fp32/times_dla.json" \
    --exportOutput="${result_dir}/fp32/output_dla.json" \
    --exportProfile="${result_dir}/fp32/profile_dla.json" \
    --exportLayerInfo="${result_dir}/fp32/layerinfo_dla.json" \
    > ${result_dir}/fp32/dump_dla.txt

for precision in fp16 int8 best
do
    # CUDA
    trtexec --onnx="$1" --saveEngine="${trt_dir}/$precision/cuda.trt" --$precision \
    --exportTimes="${result_dir}/$precision/times_cuda.json" \
    --exportOutput="${result_dir}/$precision/output_cuda.json" \
    --exportProfile="${result_dir}/$precision/profile_cuda.json" \
    --exportLayerInfo="${result_dir}/$precision/layerinfo_cuda.json" \
    > ${result_dir}/$precision/dump_cuda.txt
    # DLA
    trtexec --onnx="$1" --saveEngine="${trt_dir}/$precision/dla.trt" --$precision \
    --allowGPUFallback --useDLACore=0 \
    --exportTimes="${result_dir}/$precision/times_dla.json" \
    --exportOutput="${result_dir}/$precision/output_dla.json" \
    --exportProfile="${result_dir}/$precision/profile_dla.json" \
    --exportLayerInfo="${result_dir}/$precision/layerinfo_dla.json" \
    > ${result_dir}/$precision/dump_dla.txt
done