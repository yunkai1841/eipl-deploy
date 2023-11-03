import time
import argparse
from os import path

from common import allocate_buffers, do_inference_v2
from trt_utils import build_engine, load_engine
from data_utils import Joints, Images

from typing import Optional, Literal

models = ["sarnn", "cnnrnn", "cnnrnnln", "caebn"]


def infer(
    model: Literal["sarnn", "cnnrnn", "cnnrnnln", "caebn"] = "sarnn",
    precision: Literal["fp32", "fp16", "int8"] = "fp32",
    model_path: Optional[str] = None,
):
    # TODO: feature precision
    print("infer")
    onnx_name = f"{model}.onnx"
    engine_name = f"{model}_{precision}.trt"
    if path.exists(engine_name):
        print("load existing engine")
        engine = load_engine(engine_name)
    else:
        print("engine not found, build engine")
        engine = build_engine(onnx_name, engine_name)

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    joints = Joints()
    images = Images()

    n_loop = len(images)

    for loop_ct in range(n_loop):
        inputs[0].host = images[loop_ct]
        inputs[1].host = joints[loop_ct]

        # inference
        t1 = time.time()
        result = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        t2 = time.time()
        print("inference time:{}".format(t2 - t1))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", choices=models, default="sarnn")
    args.add_argument("--int8", action="store_true")
    args.add_argument("--fp16", action="store_true")
    args = args.parse_args()

    precision = "fp32"
    if args.int8:
        precision = "int8"
    elif args.fp16:
        precision = "fp16"

    infer(args.model, precision)
