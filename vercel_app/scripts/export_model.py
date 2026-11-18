"""Export the project's trained checkpoint for ONNX Runtime inference."""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))
from pcb.engine import Detector

torch.set_num_threads(2)
detector = Detector.load(PROJECT / 'outputs/yolov8n_gpu_full_01/weights/best.pt', 'cpu')
model = detector.model.eval()
head = model.model[-1]
head.export = True
head.format = 'onnx'
head.dynamic = False
destination = ROOT / 'model/pcb.onnx'
destination.parent.mkdir(exist_ok=True)
example = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    model(example)
    torch.onnx.export(model, example, str(destination), input_names=['images'],
                      output_names=['predictions'], opset_version=17,
                      dynamo=False, do_constant_folding=True)
print(f'Exported {destination.stat().st_size:,} bytes')
