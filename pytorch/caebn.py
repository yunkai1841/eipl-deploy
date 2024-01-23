"""
Profile CAEBN inference
Inference with pretrained weight
No write result only inference
Download pretrained before run this script
"""
import os
import torch
import time
import json
import numpy as np
import argparse
from memory_profiler import profile

from models.caebn import CAEBN


argparser = argparse.ArgumentParser()
# argparser.add_argument(
#     "--input_param",
#     type=float,
#     default=1.0,
#     help="input mix parameter in range [0.0-1.0]",
# )
argparser.add_argument("--device", "-d", choices=["cuda", "cpu"], default="cuda")
argparser.add_argument("--loop", "-l", type=int, help="number of inference loop")
argparser.add_argument(
    "--profile", "-p", action="store_true", help="enable memory profiler"
)
argparser.add_argument(
    "--parallel", action="store_true", help="enable parallel inference"
)
args = argparser.parse_args()

if args.device == "cuda" and not torch.cuda.is_available():
    print("cuda is not available, use cpu instead")
    device = torch.device("cpu")
else:
    device = torch.device(args.device)
print("device:{}".format(device))

weight_file = "./downloads/airec/pretrained/CAEBN/model.pth"
params_file = "./downloads/airec/pretrained/CAEBN/args.json"
data_dir = "./downloads/airec/grasp_bottle"
test_data_dir = "./downloads/airec/grasp_bottle/test"
test_data_index = 0

# load parameters
with open(params_file) as f:
    params = json.load(f)

# load test data
images = np.load(os.path.join(test_data_dir, "images.npy"))[test_data_index]
images = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).to(device)
print("data loaded")

# load model
model = CAEBN(feat_dim=params["feat_dim"]).to(device)

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


def inference():
    # warm up
    for loop_ct in range(3):
        img_t = images[loop_ct].unsqueeze(0)
        img_t = img_t / 255.0
        with torch.inference_mode():
            model(img_t)

    for loop_ct in range(nloop):
        # prepare data
        img_t = images[loop_ct].unsqueeze(0)
        img_t = img_t / 255.0

        # if PyTorch<1.9 use torch.no_grad() instead
        with torch.inference_mode():
            start_time = time.perf_counter()
            model(img_t)
            end_time = time.perf_counter()
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


def parallel_inference():
    img = images / 255.0
    with torch.inference_mode():
        model(img)  # warm up
        start_time = time.perf_counter()
        model(img)
        end_time = time.perf_counter()
    elapsed = end_time - start_time

    print("\n\nsummary=====================================")
    print(f"device={device}")
    print(f"inference time={elapsed}")
    print(f"total images={len(images)}")
    print(f"fps={len(images) / elapsed}")


if args.parallel:
    parallel_inference()
    exit()

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
