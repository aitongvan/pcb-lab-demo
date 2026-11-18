"""Check that ONNX deployment preserves the original detector's predictions."""
from pathlib import Path
import argparse
import io
import json
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from inference import predict
from pcb.engine import Detector

torch.set_num_threads(2)
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['yolov8n', 'yolo11n'], default='yolov8n')
parser.add_argument('--checkpoint', type=Path)
args = parser.parse_args()
if args.model == 'yolo11n' and not args.checkpoint:
    parser.error('--checkpoint is required for YOLO11n verification')
checkpoint = args.checkpoint or ROOT.parent / 'outputs/yolov8n_gpu_full_01/weights/best.pt'
detector = Detector.load(checkpoint, 'cpu')
assert detector.arch == args.model and detector.mode == 'test', 'Wrong architecture or input mode'
rows = json.loads((ROOT.parent / 'data/processed/manifest.json').read_text(encoding='utf-8'))
scenarios = []
for row in rows:
    with Image.open(ROOT.parent / row['tested']) as source:
        image = source.convert('RGB')
    scenarios.extend([(row['id'], image, .05, False), (row['id'], image, .25, False)])
scenarios.append(('rectangular', image.resize((960, 480)), .25, False))
scenarios.append(('contrast', image, .25, True))
scenarios.append(('tiled', image.resize((1340, 760)), .25, False))

for name, image, confidence, enhance in scenarios:
    gray = image.convert('L')
    if enhance:
        from PIL import ImageOps
        gray = ImageOps.autocontrast(gray, cutoff=1)
    expected, _, tiles = detector.predict_image(np.asarray(gray), confidence)
    actual = predict(image, confidence, enhance, args.model)
    assert len(expected) == len(actual['predictions']), (name, len(expected), len(actual['predictions']))
    assert tiles == actual['tiles_scanned']
    unmatched = list(actual['predictions'])
    for prediction in expected:
        matching = [p for p in unmatched if p['class_id'] == prediction['class_id']]
        assert matching, (name, prediction)
        found = min(matching, key=lambda p: np.linalg.norm(np.array(p['bbox_xyxy']) - prediction['bbox_xyxy']))
        np.testing.assert_allclose(found['bbox_xyxy'], prediction['bbox_xyxy'], atol=.05, rtol=0)
        np.testing.assert_allclose(found['score'], prediction['score'], atol=.0001, rtol=0)
        unmatched.remove(found)
    print(name, confidence, len(expected), 'detections matched')

print(f'All {len(scenarios)} ONNX and PyTorch prediction comparisons passed for {args.model}.')
