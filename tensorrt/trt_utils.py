import tensorrt as trt
import numpy as np
from os import path, makedirs, remove
import shutil

# import random

from data_utils import Joints, Images
from common import HostDeviceMem, allocate_buffers, do_inference_v2


# tensorrt log level
LOG_LEVEL = trt.Logger.INFO
# LOG_LEVEL = trt.Logger.VERBOSE # for debug


def build_engine(
    model, onnx_file_path, engine_file_path, precision="fp32", rm_cache=False
):
    """
    Build TensorRT engine from ONNX file and save it.
    """
    logger = trt.Logger(LOG_LEVEL)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_file_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        raise Exception("Failed to parse the ONNX file.")

    config = builder.create_builder_config()
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError(
                "FP16 mode requested on a platform that doesn't support it!"
            )
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError(
                "INT8 mode requested on a platform that doesn't support it!"
            )
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = Int8Calibrator(model=model, rm_cache=rm_cache)
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


def load_engine(engine_file_path):
    """
    Load TensorRT engine.
    """
    logger = trt.Logger(LOG_LEVEL)
    runtime = trt.Runtime(logger)
    with open(engine_file_path, "rb") as f:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, model: str = "sarnn", rm_cache: bool = False):
        super(Int8Calibrator, self).__init__()
        self.model = model
        calibration_data_dir = f"calibration/{model}"
        self.cache_file = path.join(calibration_data_dir, "calibration.cache")
        if rm_cache and path.exists(self.cache_file):
            remove(self.cache_file)
        if rm_cache and path.exists(calibration_data_dir):
            shutil.rmtree(calibration_data_dir)
        if not path.exists(calibration_data_dir):
            gen_calibration_data(model=model, dataset_index=0)
        self.images = np.load(f"{calibration_data_dir}/image.npy")
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            self.joints = np.load(f"{calibration_data_dir}/joint.npy")
            self.state_h = np.load(f"{calibration_data_dir}/state_h.npy")
            self.state_c = np.load(f"{calibration_data_dir}/state_c.npy")
        self.batch_size = 1
        self.input_shapes = (
            {
                "i.image": (3, 128, 128),
                "i.joint": (8,),
                "i.state_h": (50,),
                "i.state_c": (50,),
            }
            if model in ["sarnn", "cnnrnn", "cnnrnnln"]
            else {"i.image": (3, 128, 128)}
        )
        self.names = set(self.input_shapes.keys())
        self.host_device_mem_dic = {}
        for name in self.names:
            shape = self.input_shapes[name]
            size = trt.volume(shape)
            self.host_device_mem_dic[name] = HostDeviceMem(size, np.dtype(np.float32))
        self.batch_idx = 0
        self.max_batch_idx = len(self.images)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.batch_idx >= self.max_batch_idx:
            return None
        else:
            image = self.images[self.batch_idx]
            self.host_device_mem_dic["i.image"].host = image
            if self.model in ["sarnn", "cnnrnn", "cnnrnnln"]:
                joint = self.joints[self.batch_idx]
                state_h = self.state_h[self.batch_idx]
                state_c = self.state_c[self.batch_idx]
                self.host_device_mem_dic["i.joint"].host = joint
                self.host_device_mem_dic["i.state_h"].host = state_h
                self.host_device_mem_dic["i.state_c"].host = state_c
            self.batch_idx += 1
            if self.model in ["sarnn", "cnnrnn", "cnnrnnln"]:
                return [self.host_device_mem_dic[name].device for name in names]
            else:
                return [self.host_device_mem_dic["i.image"].device]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def gen_calibration_data(model="sarnn", dataset_index=0):
    """
    Generate calibration data.
    """
    onnx_name = f"models/{model}.onnx"
    engine_name = f"models/{model}_fp32.trt"
    if path.exists(engine_name):
        engine = load_engine(engine_name)
    else:
        engine = build_engine(model, onnx_name, engine_name, precision="fp32")

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream, input_names, output_names = allocate_buffers(
        engine
    )
    input_names = {name: i for i, name in enumerate(input_names)}
    output_names = {name: i for i, name in enumerate(output_names)}

    np_dtype = np.float32
    joints = Joints(dataset_index=dataset_index, dtype=np_dtype)
    images = Images(dataset_index=dataset_index, dtype=np_dtype)
    lstm_state_h = np.zeros(50, order="C").astype(np_dtype)
    lstm_state_c = np.zeros(50, order="C").astype(np_dtype)

    n_loop = len(images)
    # ? How to choose calibration data keeping the distribution of the whole dataset?
    # ? or just use all data
    # save_index = random.sample(range(10, n_loop - 10), 10)
    # print(f"calibration data index: {save_index}")
    save_data = {
        "image": [],
        "joint": [],
        "state_h": [],
        "state_c": [],
    }
    for loop_ct in range(n_loop):
        # if loop_ct in save_index:
        save_data["image"].append(images[loop_ct])
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            save_data["joint"].append(joints[loop_ct])
            save_data["state_h"].append(lstm_state_h)
            save_data["state_c"].append(lstm_state_c)

        inputs[input_names["i.image"]].host = images[loop_ct]
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            inputs[input_names["i.joint"]].host = joints[loop_ct]
            inputs[input_names["i.state_h"]].host = lstm_state_h
            inputs[input_names["i.state_c"]].host = lstm_state_c

        # inference
        result = do_inference_v2(
            context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )

        # update lstm state
        if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
            lstm_state_h = result[output_names["o.state_h"]].copy()
            lstm_state_c = result[output_names["o.state_c"]].copy()

    # save calibration data
    dir_name = f"calibration/{model}"
    if not path.exists(dir_name):
        makedirs(dir_name)

    np.save(f"{dir_name}/image.npy", np.array(save_data["image"]))
    if model in ["sarnn", "cnnrnn", "cnnrnnln"]:
        np.save(f"{dir_name}/joint.npy", np.array(save_data["joint"]))
        np.save(f"{dir_name}/state_h.npy", np.array(save_data["state_h"]))
        np.save(f"{dir_name}/state_c.npy", np.array(save_data["state_c"]))
