#!/bin/bash

image_name="trt"
dockerfile_path=".devcontainer/Dockerfile"
container_name="trt"

build_docker() {
    local image_name="$1"
    local dockerfile_path="$2"
    local container_name="$3"

    # Build the Docker image
    docker build -t "$image_name" -f "$dockerfile_path" .
}

run_docker() {
    local image_name="$1"
    local container_name="$2"

    # Run the Docker container and attach it
    docker run -it \
        --gpus all \
        --rm \
        --name "$container_name" "$image_name"
}

# Build the Docker image if not exists

if [[ "$(docker images -q $image_name 2> /dev/null)" == "" ]]; then
    build_docker "$image_name" "$dockerfile_path" "$container_name"
fi

# Run the Docker container

run_docker "$image_name" "$container_name"