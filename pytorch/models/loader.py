import json
import os
import torch
import models


def model_loader(model_name, device=torch.device("cpu")):
    with open(
        os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            os.path.pardir,
            "configs/model_config.json",
        )
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
    else:
        raise ValueError(f"Model {model_name} not found in models/__init__.py")

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# dummy input generator
def rand_image():
    return torch.randn(1, 3, 128, 128)


def rand_joint():
    return torch.randn(1, 8)


def rand_state():
    return tuple(torch.randn(1, 50) for _ in range(2))


def export_onnx(model_name, file):
    model = model_loader(model_name)
    if model_name == "caebn":
        dummy_input = rand_image()
    else:
        dummy_input = (rand_image(), rand_joint(), rand_state())
    input_names = ["i.image"]
    output_names = ["o.image"]
    if model_name == "sarnn":
        input_names += ["i.joint", "i.state_h", "i.state_c"]
        output_names += ["o.joint", "o.enc_pts", "o.dec_pts", "o.state_h", "o.state_c"]
    elif model_name in ["cnnrnn", "cnnrnnln"]:
        input_names += ["i.joint", "i.state_h", "i.state_c"]
        output_names += ["o.joint", "o.state_h", "o.state_c"]
    torch.onnx.export(
        model,
        dummy_input,
        file,
        input_names=input_names,
        output_names=output_names,
        verbose=True,
    )
    print(f"ONNX model exported to {file}")
