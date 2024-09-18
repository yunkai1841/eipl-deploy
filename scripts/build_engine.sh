#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# use function from docker_utils.sh
source $SCRIPT_DIR/docker_utils.sh

# build engine

onnx_file=$1
engine_file=$2

run_docker \
    $l4t_trt_docker_img \
    $l4t_trt_docker_img_tag \
    /usr/src/tensorrt/bin/trtexec \
    "--onnx=$onnx_file --saveEngine=$engine_file --profilingVerbosity=detailed"
