"""
Profile SARNN inference
Inference with pretrained weight
No write result only inference
Download pretrained before run this script
"""
import os
import torch
import time
import json
import sys
import numpy as np
import argparse
from memory_profiler import profile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.sarnn import SARNN


argparser = argparse.ArgumentParser()
# argparser.add_argument("--input_param", type=float, default=1.0)
argparser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
argparser.add_argument("--loop", type=int, help="number of inference loop")
argparser.add_argument("--profile", action="store_true")
args = argparser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    print("cuda is not available, use cpu instead")
    device = torch.device("cpu")
else:
    device = torch.device(args.device)
print("device:{}".format(device))

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

# alanysis
time_list = []
if args.loop is not None:
    nloop = args.loop
else:
    nloop = len(images)


# Inference
def inference():
    state = None
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
        _, _, _, _, state = model(img_t, joint_t, state)
        end_time = time.time()
        elapsed = end_time - start_time
        time_list.append(elapsed)

        if loop_ct == 0:
            # first loop is slower than others
            # so remove first loop from avg time
            print(f"inference time={elapsed}")
        else:
            print(
                f"inference time={elapsed}, avg time={sum(time_list[1:]) / len(time_list[1:])}"
            )

if args.profile:
    inference = profile(inference)()
else:
    inference()

# print summary
print("\n\nsummary=====================================")
print(f"device={device}")
print(f"total loop={nloop}")

print(f"total inference time={sum(time_list)}")
print(f"avg inference time={sum(time_list) / len(time_list)}")
print(f"fps={len(time_list) / sum(time_list)}")

print("\nsummary without first loop==================")
print(f"total inference time={sum(time_list[1:])}")
print(f"avg inference time={sum(time_list[1:]) / len(time_list[1:])}")
print(f"fps={len(time_list[1:]) / sum(time_list[1:])}")
