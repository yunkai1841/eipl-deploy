import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.loader import export_onnx

export_onnx("sarnn", "sarnn.onnx")
export_onnx("cnnrnn", "cnnrnn.onnx")
export_onnx("cnnrnnln", "cnnrnnln.onnx")
export_onnx("caebn", "caebn.onnx")

