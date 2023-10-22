import json
import os
import torch
import models


def model_loader(model_name, device):
    with open(
        os.path.join(os.path.dirname(__file__), "../configs/model_config.json")
    ) as f:
        model_config = json.load(f)
    if model_name not in model_config:
        raise ValueError(f"Model {model_name} not found in configs/model_config.json")
    model_info = model_config[model_name]
    base = model_info["base"]
    weight = model_info["weight"]
    args = model_info["args"]

    ckpt = torch.load(os.path.join(base, weight), map_location=device)

    with open(os.path.join(base, args)) as f:
        params = json.load(f)

    if model_name == "sarnn":
        model = models.SARNN(
            rec_dim=params["rec_dim"],
            joint_dim=8,
            k_dim=params["k_dim"],
            heatmap_size=params["heatmap_size"],
            temperature=params["temperature"],
        ).to(device)
    elif model_name == "cnnrnn":
        model = models.CNNRNN(
            rec_dim=params["rec_dim"], joint_dim=8, feat_dim=params["feat_dim"]
        ).to(device)
    elif model_name == "cnnrnnln":
        model = models.CNNRNNLN(
            rec_dim=params["rec_dim"], joint_dim=8, feat_dim=params["feat_dim"]
        ).to(device)
    elif model_name == "caebn":
        model = models.CAEBN(feat_dim=params["feat_dim"])


    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model
