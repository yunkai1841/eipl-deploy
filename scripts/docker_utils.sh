#!/bin/bash

trt_docker_img="nvcr.io/nvidia/tensorrt"
trt_docker_img_tag="23.01-py3"

l4t_trt_docker_img="nvcr.io/nvidia/l4t-tensorrt"
l4t_trt_docker_img_tag="r8.5.2.2-devel"

function run_docker() {
    local docker_img=$1
    local docker_img_tag=$2
    local docker_cmd=$3
    local docker_args=$4

    docker run --rm -it --runtime nvidia \
        --network host \
        --volume $(pwd):/workspace \
        --workdir /workspace \
        $docker_img:$docker_img_tag $docker_cmd $docker_args
}

function build_engine() {
    local model_file=$1
    local engine_file=$2
    local trtexec_args=$3

    run_docker $trt_docker_img $trt_docker_img_tag trtexec "--onnx=$model_file --saveEngine=$engine_file $trtexec_args"
}

function run_engine() {
    local engine_file=$1
    local trtexec_args=$2

    run_trtexec $engine_file "$trtexec_args"
}

function enter_docker_trt() {
    run_docker $trt_docker_img $trt_docker_img_tag bash
}

function enter_docker_l4t_trt() {
    # l4t only supports arm64
    if [ "$(uname -m)" != "aarch64" ]; then
        echo "This script is for arm64 only"
        exit 1
    fi

    run_docker $l4t_trt_docker_img $l4t_trt_docker_img_tag bash
}
