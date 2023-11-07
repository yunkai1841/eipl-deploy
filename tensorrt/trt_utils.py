import tensorrt as trt
import numpy as np
import time

import common


# tensorrt log level
LOG_LEVEL = trt.Logger.INFO
# LOG_LEVEL = trt.Logger.VERBOSE # for debug

def build_engine(onnx_file_path, engine_file_path):
    """
    Build TensorRT engine from ONNX file and save it.
    """
    logger = trt.Logger(LOG_LEVEL)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    success = parser.parse_from_file(onnx_file_path)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))
    if not success:
        raise Exception('Failed to parse the ONNX file.')
    
    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20) # 1 MiB
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_file_path, 'wb') as f:
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
    with open(engine_file_path, 'rb') as f:
        serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine
