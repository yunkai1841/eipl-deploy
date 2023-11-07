import time
import argparse
from os import path
import numpy as np

from common import allocate_buffers, do_inference_v2
from trt_utils import build_engine, load_engine
from data_utils import Joints, Images
from logger import (
    TimeResultShower,
    PowerLogger,
    InferenceResultShower,
    sarnn_image_postprocess,
)

from typing import Optional, Literal

models = ["sarnn", "cnnrnn", "cnnrnnln", "caebn"]


def infer(
    model: Literal["sarnn", "cnnrnn", "cnnrnnln", "caebn"] = "sarnn",
    precision: Literal["fp32", "fp16", "int8"] = "fp32",
    model_path: Optional[str] = None,
    warmup_loops: int = 1000,
    measure_power: bool = False,
    measure_time: bool = True,
    show_result: bool = False, # use false in server
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
    inputs, outputs, bindings, stream, input_names, output_names = allocate_buffers(engine)
    # make input_names and output_names dict
    input_names = {name: i for i, name in enumerate(input_names)}
    output_names = {name: i for i, name in enumerate(output_names)}

    joints = Joints()
    images = Images()
    lstm_state_h = np.zeros(50, order="C").astype(np.float32)
    lstm_state_c = np.zeros(50, order="C").astype(np.float32)

    if measure_power:
        power_logger = PowerLogger()
    if measure_time:
        time_shower = TimeResultShower()
    inference_shower = InferenceResultShower(
        image_postprocess=sarnn_image_postprocess,
        joint_postprocess=joints.denormalize,
    )

    # warmup
    print(f"warmup {warmup_loops} loops")
    for _ in range(warmup_loops):
        inputs[input_names["i.image"]].host = images.random()
        inputs[input_names["i.joint"]].host = joints.random()
        inputs[input_names["i.state_h"]].host = np.random.random(50).astype(np.float32)
        inputs[input_names["i.state_c"]].host = np.random.random(50).astype(np.float32)
        do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
    time.sleep(3)

    # inference
    print("inference start")
    if measure_power:
        power_logger.start_measure()

    n_loop = len(images)
    for loop_ct in range(n_loop):
        inputs[input_names["i.image"]].host = images[loop_ct]
        inputs[input_names["i.joint"]].host = joints[loop_ct]
        inputs[input_names["i.state_h"]].host = lstm_state_h
        inputs[input_names["i.state_c"]].host = lstm_state_c

        # inference
        t1 = time.time()
        result = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        t2 = time.time()
        elapsed = t2 - t1
        print("inference time:{}".format(elapsed))

        # postprocess
        if measure_time:
            time_shower.append(elapsed)
        inference_shower.append(
            images[loop_ct],
            result[output_names["o.image"]],
            joints[loop_ct],
            result[output_names["o.joint"]],
            result[output_names["o.enc_pts"]],
            result[output_names["o.dec_pts"]],
        )

        # update lstm state
        lstm_state_h = result[output_names["o.state_h"]].copy()
        lstm_state_c = result[output_names["o.state_c"]].copy()

    if measure_power:
        power_logger.stop_measure()
        power_logger.summary(save="power.txt")
        power_logger.save_csv(save="power.csv")
        power_logger.plot(show=show_result, save="power.png")

    if measure_time:
        time_shower.summary(save="time.txt")
        time_shower.save_csv(save="time.csv")
        time_shower.plot(show=show_result, save="time.png")

    inference_shower.plot(show=show_result, save="result.mp4")


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
