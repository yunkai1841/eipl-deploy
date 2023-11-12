import tensorrt as trt


# tensorrt log level
LOG_LEVEL = trt.Logger.INFO
# LOG_LEVEL = trt.Logger.VERBOSE # for debug


def build_engine(onnx_file_path, engine_file_path, precision="fp32"):
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
        config.set_flag(trt.BuilderFlag.INT8)
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
