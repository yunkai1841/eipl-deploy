# EIPL tensorrt inference

## Setup

1. download pretrained weights [scripts/downloader.py](scripts/downloader.py)
2. create onnx models [pytorch/export_onnx.py](pytorch/export_onnx.py)
3. install requirements
```bash
sudo apt install python3-libnvinfer python3-libnvinfer-dev
sudo pip install jetson_stats
pip install cuda-python
```

## Run inference

```bash
python tensorrt/infer.py
```
