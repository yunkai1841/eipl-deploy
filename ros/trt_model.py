import tensorrt as trt
import numpy as np
import time

from common import allocate_buffers, do_inference_v2


class SARNNTRT:
    def __init__(self, engine_path, print_time=False):
        self.engine_path = engine_path
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream, input_names, output_names = allocate_buffers(
            self.engine
        )
        self.input_names = {name: i for i, name in enumerate(input_names)}
        self.output_names = {name: i for i, name in enumerate(output_names)}
        self.print_time = print_time

    def load_engine(self, engine_path):
        with open(engine_path, "rb") as f, trt.Runtime(
            trt.Logger(trt.Logger.WARNING)
        ) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __call__(self, img, joint, state_h, state_c):
        self.inputs[self.input_names["i.image"]].host = img
        self.inputs[self.input_names["i.joint"]].host = joint

        # TODO(performance): keep rnn state in device memory
        self.inputs[self.input_names["i.state_h"]].host = state_h
        self.inputs[self.input_names["i.state_c"]].host = state_c

        # TODO(performance): separate inference and memory transfer
        t1 = time.perf_counter()
        result = do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream
        )
        t2 = time.perf_counter()
        if self.print_time:
            print("[TRT] Memory transfer + Inference time: {} (ms)", (t2 - t1) * 1000)

        return {
            result[self.output_names["o.image"]],
            result[self.output_names["o.joint"]],
            result[self.output_names["o.enc_pts"]],
            result[self.output_names["o.dec_pts"]],
            result[self.output_names["o.state_h"]],
            result[self.output_names["o.state_c"]],
        }
