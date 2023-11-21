from models.loader import export_onnx
import os

save_dir = os.path.join(os.path.dirname(__file__), "../models")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for model in ["sarnn", "cnnrnn", "cnnrnnln", "caebn"]:
    export_onnx(model, os.path.join(save_dir, f"{model}.onnx"))

# visualize onnx
# import netron

# netron.start("sarnn.onnx")
# netron.start("cnnrnn.onnx")
# netron.start("cnnrnnln.onnx")
# netron.start("caebn.onnx")
