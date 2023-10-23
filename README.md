# eipl-deploy
Minimized ogata-lab/eipl code, which aim to deploy on edge devices

## Tested environment
- Jetson Xavier NX
- JetPack 5.1.1
- See [requrements_jetson.txt](requirements_jetson.txt)

## Installation
Install python packages using the following command.
```bash
pip install -r requirements_jetson.txt
```

Download sample data and pretrained weights using the following command.
```bash
python scripts/downloader.py
```

## Supported Models
- SARNN
- CNNRNN
- CNNRNNLN
- CAELN (no RNN)
