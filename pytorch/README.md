# eipl pytorch inference

## Inference scripts
- sarnn.py
- cnnrnn.py
- cnnrnnln.py
- caebn.py

```bash
python pytorch/sarnn.py --help
python pytorch/cnnrnn.py --help
python pytorch/cnnrnnln.py --help
python pytorch/caebn.py --help
```

**Note**: download pretrained weights before inference.
```bash
python scripts/downloader.py
```

## Export ONNX models
```bash
python pytorch/export_onnx.py
```

