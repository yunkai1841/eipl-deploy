![logo](https://raw.githubusercontent.com/ogata-lab/eipl-docs/b50ecbbbc026474fa385636b73fd9ce2dcd8381a/top/resources/logo.svg)

# eipl-deploy
This repository is optimized inference code for [eipl](https://github.com/ogata-lab/eipl/)

## Tested environment
- Jetson Xavier NX
- JetPack 5.1.1
- See [requrements_jetson.txt](requirements_jetson.txt)

## Installation
Install python packages using the following command.
```bash
pip install -r requirements_jetson.txt
```

Install packages for tensorrt inference.
You need to install those packages with sudo.
```bash
sudo apt install python3-libnvinfer python3-libnvinfer-dev ffmpeg
sudo pip install jetson-stats
```

Download sample data and pretrained weights using the following command.
```bash
python scripts/downloader.py
```

Convert pretrained weights to onnx format using the following command.
```bash
python pytorch/export_onnx.py
```

## Supported Models
- SARNN
- CNNRNN
- CNNRNNLN
- **Experimental** CAEBN (no RNN)
  - Do not work with Int8 TensorRT inference
