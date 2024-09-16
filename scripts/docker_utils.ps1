$trt_docker_img = "nvcr.io/nvidia/tensorrt"
$trt_docker_img_tag = "23.01-py3"

function Run-Docker {
    param (
        [string]$docker_img,
        [string]$docker_img_tag,
        [string]$docker_cmd,
        [string]$docker_args
    )

    # if docker_args is not provided
    if (-not $docker_args) {
        docker run --rm -it --runtime nvidia `
            --network host `
            --volume ${PWD}:"/workspace" `
            --workdir /workspace `
            ${docker_img}:$docker_img_tag $docker_cmd
    } else {
        docker run --rm -it --runtime nvidia `
            --network host `
            --volume ${PWD}:"/workspace" `
            --workdir /workspace `
            ${docker_img}:$docker_img_tag $docker_cmd $docker_args
    }
}

function Build-Engine {
    param (
        [string]$model_file,
        [string]$engine_file,
        [string]$trtexec_args
    )

    Run-Docker $trt_docker_img $trt_docker_img_tag "trtexec" "--onnx=$model_file --saveEngine=$engine_file $trtexec_args"
}

function Run-Engine {
    param (
        [string]$engine_file,
        [string]$trtexec_args
    )

    Run-Docker $trt_docker_img $trt_docker_img_tag "trtexec" "--loadEngine=$engine_file $trtexec_args"
}

function Enter-Docker-TRT {
    Run-Docker $trt_docker_img $trt_docker_img_tag "bash"
}
