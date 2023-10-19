"""
export SARNN model to ONNX
"""
import os
import torch
import json
import sys
import numpy as np
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sarnn import SARNN


argparser = argparse.ArgumentParser()
argparser.add_argument("file", help="output file name")
args = argparser.parse_args()

device = torch.device("cpu")

weight_file = "./downloads/airec/pretrained/SARNN/model.pth"
params_file = "./downloads/airec/pretrained/SARNN/args.json"
data_dir = "./downloads/airec/grasp_bottle"
test_data_dir = "./downloads/airec/grasp_bottle/test"
test_data_index = 0

# load parameters
with open(params_file) as f:
    params = json.load(f)

# load test data
minmax = (params["vmin"], params["vmax"])
images = np.load(os.path.join(test_data_dir, "images.npy"))[test_data_index]
images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).to(device)
joints = np.load(os.path.join(test_data_dir, "joints.npy"))[test_data_index]
joints = torch.from_numpy(joints).to(device)
joint_bounds = np.load(os.path.join(data_dir, "joint_bounds.npy"))
joint_bounds = torch.from_numpy(joint_bounds).to(device)
print("data loaded")

# load model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=8,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
).to(device)

# load weight
ckpt = torch.load(weight_file, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print("model loaded")

# export to ONNX
img_size = 128 # 128x128x3
dummy_input = (
    torch.randn(1, 3, img_size, img_size),
    torch.randn(1, 8),
    tuple(torch.randn(1, params["rec_dim"]) for _ in range(2)),
)
torch.onnx.export(
    model,
    dummy_input,
    args.file,
    verbose=True,
    input_names=["image", "joint", "state1", "state2"],
    output_names=["image", "joint", "ect_pts", "dec_pts", "state"],
)
