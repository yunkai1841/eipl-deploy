# EIPL tensorrt inference

## Run inference

Run inference CUI
```bash
python tensorrt/infer.py
```

Run inference with web UI
```bash
python tensorrt/launcher.py
```

Run all models and precision
```bash
./scripts/run_all_trt.sh
```

## Usage
```bash
usage: infer.py [-h] [--model {sarnn,cnnrnn,cnnrnnln,caebn}] [--int8] [--fp16] [--best] [--dataset-index DATASET_INDEX] [--sleep-after-warmup SLEEP_AFTER_WARMUP] [--power]
                [--model-path MODEL_PATH] [--save-output] [--no-video] [--clear-result] [--force-build] [--gen-calibration-data]

sarnn: Spatial Attention RNN
cnnrnn: CNN + RNN
cnnrnnln: CNN + RNN + LayerNorm
caebn: CNN + AE + BatchNorm

optional arguments:
  -h, --help            show this help message and exit
  --model {sarnn,cnnrnn,cnnrnnln,caebn}
  --int8
  --fp16
  --best
  --dataset-index DATASET_INDEX
  --sleep-after-warmup SLEEP_AFTER_WARMUP
  --power
  --model-path MODEL_PATH
  --save-output
  --no-video
  --clear-result        clear result files(*.txt, *.csv, *.png, *.mp4)
  --force-build         ignore cached engine, and build new engine
  --gen-calibration-data
                        generate calibration data for int8 mode
```
