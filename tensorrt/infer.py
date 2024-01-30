import time
import argparse
from os import path, remove
import numpy as np

from common import allocate_buffers, do_inference_v2
from trt_utils import build_engine, load_engine, gen_calibration_data
from data_utils import Joints, Images
from logger import (
    TimeResultShower,
    PowerLogger,
    InferenceResultShower,
    sarnn_image_postprocess,
)

from typing import Optional, Literal

models = ["sarnn", "cnnrnn", "cnnrnnln", "caebn"]

models_description = {
    "sarnn": "Spatial Attention RNN",
    "cnnrnn": "CNN + RNN",
    "cnnrnnln": "CNN + RNN + LayerNorm",
    "caebn": "CNN + AE + BatchNorm",
}

models_io = {
    "sarnn": {
        "input": ["i.image", "i.joint", "i.state_h", "i.state_c"],
        "output": [
            "o.image",
            "o.joint",
            "o.enc_pts",
            "o.dec_pts",
            "o.state_h",
            "o.state_c",
        ],
    },
    "cnnrnn": {
        "input": ["i.image", "i.joint", "i.state_h", "i.state_c"],
        "output": ["o.image", "o.joint", "o.state_h", "o.state_c"],
    },
    "cnnrnnln": {
        "input": ["i.image", "i.joint", "i.state_h", "i.state_c"],
        "output": ["o.image", "o.joint", "o.state_h", "o.state_c"],
    },
    "caebn": {
        "input": ["i.image"],
        "output": ["o.image"],
    },
}


def infer(
    model: Literal["sarnn", "cnnrnn", "cnnrnnln", "caebn"] = "sarnn",
    precision: Literal["fp32", "fp16", "int8", "best"] = "fp32",
    dataset_index: int = 0,
    model_path: Optional[str] = None,
    warmup_iter: int = 1000,
    sleep_after_warmup: float = 0.0,
    measure_power: bool = False,
    measure_time: bool = True,
    show_result: bool = False,  # use false in server
    force_build_engine: bool = False,
    save_output_video: bool = True,
    save_output_numpy: bool = False,
    progress_logger: Optional[object] = lambda _, txt: print(txt),
):
    if model_path is None:
        onnx_name = f"models/{model}.onnx"
        engine_name = f"models/{model}_{precision}.trt"
        if path.exists(engine_name) and not force_build_engine:
            progress_logger(0.1, "loading existing engine")
            engine = load_engine(engine_name)
        else:
            progress_logger(0.1, "building engine, this may take a while")
            engine = build_engine(
                model,
                onnx_name,
                engine_name,
                precision=precision,
                rm_cache=force_build_engine,
            )
    else:
        model_type = path.splitext(model_path)[1]
        if model_type == ".onnx":
            engine_name = f"{path.splitext(model_path)[0]}_{precision}.trt"
            if path.exists(engine_name) and not force_build_engine:
                progress_logger(0.1, "loading existing engine")
                engine = load_engine(engine_name)
            else:
                progress_logger(0.1, "building engine, this may take a while")
                engine = build_engine(
                    model,
                    model_path,
                    engine_name,
                    precision=precision,
                    rm_cache=force_build_engine,
                )
        elif model_type == ".trt":
            engine = load_engine(model_path)
        else:
            raise ValueError(f"unknown model type {model_type}")

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream, input_names, output_names = allocate_buffers(
        engine
    )
    # make input_names and output_names dict
    input_names = {name: i for i, name in enumerate(input_names)}
    output_names = {name: i for i, name in enumerate(output_names)}

    np_dtype = np.float32
    joints = Joints(dataset_index=dataset_index, dtype=np_dtype)
    images = Images(dataset_index=dataset_index, dtype=np_dtype)
    lstm_state_h = np.zeros(50, order="C").astype(np_dtype)
    lstm_state_c = np.zeros(50, order="C").astype(np_dtype)

    if measure_power:
        power_logger = PowerLogger()
    if measure_time:
        time_shower = TimeResultShower()
    inference_shower = InferenceResultShower(
        model=model,
        image_postprocess=sarnn_image_postprocess,
        joint_postprocess=joints.denormalize
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]
        else lambda x: x,
    )
    # warmup
    progress_logger(0.4, f"warming up {warmup_iter} loops")
    for _ in range(warmup_iter):
        inputs[input_names["i.image"]].host = images.random()
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            inputs[input_names["i.joint"]].host = joints.random()
            inputs[input_names["i.state_h"]].host = np.random.random(50).astype(
                np_dtype
            )
            inputs[input_names["i.state_c"]].host = np.random.random(50).astype(
                np_dtype
            )
        do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
    if sleep_after_warmup > 0:
        time.sleep(sleep_after_warmup)

    # inference
    progress_logger(0.5, "main inference")
    if measure_power:
        power_logger.start_measure()

    n_loop = len(images)
    pre_t = time.perf_counter()
    for loop in range(10000):
        loop_ct = loop % n_loop
        inputs[input_names["i.image"]].host = images[loop_ct]
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            inputs[input_names["i.joint"]].host = joints[loop_ct]
            inputs[input_names["i.state_h"]].host = lstm_state_h
            inputs[input_names["i.state_c"]].host = lstm_state_c

        # inference
        t1 = time.perf_counter()
        result = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        t2 = time.perf_counter()
        elapsed = t2 - t1
        print("inference time:{}".format(elapsed))

        # postprocess
        if measure_time:
            time_shower.append(elapsed)
        empty = np.zeros(0, dtype=np_dtype)
        inference_shower.append(
            images[loop_ct],
            result[output_names["o.image"]],
            joints[loop_ct] if model in ["sarnn", "cnnrnn", "cnnrnnln"] else empty,
            result[output_names["o.joint"]]
            if model in ["sarnn", "cnnrnn", "cnnrnnln"]
            else empty,
            result[output_names["o.enc_pts"]] if model in ["sarnn"] else empty,
            result[output_names["o.dec_pts"]] if model in ["sarnn"] else empty,
            elapsed,
        )

        # update lstm state
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            lstm_state_h = result[output_names["o.state_h"]].copy()
            lstm_state_c = result[output_names["o.state_c"]].copy()
    post_t = time.perf_counter()
    print("total time:{}".format(post_t - pre_t))

    progress_logger(0.6, "collecting result")

    if measure_power:
        power_logger.stop_measure()
        power_logger.summary(save="power.txt", latency_frame = (post_t - pre_t) / 10000)
        power_logger.save_csv(save="power.csv")
        power_logger.plot(show=show_result, save="power.png")

    # if measure_time:
    #     time_shower.summary(save="time.txt")
    #     time_shower.save_csv(save="time.csv")
    #     time_shower.plot(show=show_result, save="time.png")

    # if save_output_numpy:
    #     # progress_logger(0.7, "saving output numpy")
    #     inference_shower.save_numpy(image_save="image.npy", joint_save="joint.npy")
    # if save_output_video:
    #     progress_logger(0.7, "encoding result video, this may take a while")
    #     inference_shower.plot(show=show_result, save="result.mp4")
    # inference_shower.summary(save="result.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", choices=models, default="sarnn")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--sleep-after-warmup", type=float, default=0.0)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--save-output", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--clear-result",
        action="store_true",
        help="clear result files(*.txt, *.csv, *.png, *.mp4)",
    )
    parser.add_argument(
        "--force-build",
        action="store_true",
        help="ignore cached engine, and build new engine",
    )
    parser.add_argument(
        "--gen-calibration-data",
        action="store_true",
        help="generate calibration data for int8 mode",
    )
    # model description + \n
    parser.description = "\n".join(
        [
            f"{model}: {models_description[model]}"
            for model in models
            if model in models_description
        ]
    )
    args = parser.parse_args()

    if args.clear_result:
        result_files = [
            "power.txt",
            "power.csv",
            "power.png",
            "time.txt",
            "time.csv",
            "time.png",
            "result.mp4",
            "image.npy",
            "joint.npy",
            "result.txt",
        ]
        for file in result_files:
            if path.exists(file):
                remove(file)

    precision = "fp32"
    if args.best:
        precision = "best"
    if args.int8:
        precision = "int8"
    elif args.fp16:
        precision = "fp16"

    if args.gen_calibration_data:
        gen_calibration_data(model=args.model, dataset_index=args.dataset_index)
        exit(0)

    infer(
        args.model,
        precision,
        measure_power=True,
        dataset_index=args.dataset_index,
        sleep_after_warmup=args.sleep_after_warmup,
        force_build_engine=args.force_build,
        model_path=args.model_path,
        save_output_numpy=args.save_output,
        save_output_video=not args.no_video,
    )
