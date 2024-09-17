import tensorrt as trt

# tensorrt log level
LOG_LEVEL = trt.Logger.INFO
# LOG_LEVEL = trt.Logger.VERBOSE # for debug

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
