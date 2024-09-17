import tensorrt as trt
import numpy as np


class SARNNTRT:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

