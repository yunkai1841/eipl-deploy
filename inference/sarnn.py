"""
Inference with pretrained weight
No write result only inference
Download pretrained before run this script
"""
import os
import torch
import time
import json
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sarnn import SARNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
print(joint_bounds)
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

# Inference
img_size = 128
s1, s2 = None, None
state = None
y_image, y_joint = None, None
time_list = []
# nloop = len(images)
nloop = 50
for loop_ct in range(nloop):
    # prepare data
    img_t = images[loop_ct].unsqueeze(0)
    img_t = img_t / 255.0
    joint_t = joints[loop_ct].unsqueeze(0)
    # joint_t = normalize(joint_t, joint_bounds, minmax)
    joint_t = (joint_t - joint_bounds[0]) / (joint_bounds[1] - joint_bounds[0])

    # closed loop
    # if loop_ct > 0:
    #     img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
    #     joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint

    # inference
    start_time = time.time()
    y_image, y_joint, _, _, state = model(img_t, joint_t, state)
    end_time = time.time()
    elapsed = end_time - start_time
    time_list.append(elapsed)

    print(f"inference time={elapsed}, avg time={sum(time_list) / len(time_list)}")
    if loop_ct == 0:
        # remove first load
        time_list = []

    print("loop_ct:{}".format(loop_ct))

