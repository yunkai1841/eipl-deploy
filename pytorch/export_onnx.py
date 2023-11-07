from models.loader import export_onnx

export_onnx("sarnn", "sarnn.onnx")
export_onnx("cnnrnn", "cnnrnn.onnx")
export_onnx("cnnrnnln", "cnnrnnln.onnx")
export_onnx("caebn", "caebn.onnx")

# visualize onnx
# import netron

# netron.start("sarnn.onnx")
# netron.start("cnnrnn.onnx")
# netron.start("cnnrnnln.onnx")
# netron.start("caebn.onnx")
