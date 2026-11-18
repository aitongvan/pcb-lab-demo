"""Export and verify a fully trained, single-image PCB YOLO11n checkpoint."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from pcb.engine import Detector
from pcb.data import manifest, load_pair, resize_input
from pcb.common import project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='yolo11n_single_full_01')
    args = parser.parse_args()
    run = PROJECT / 'outputs' / args.run
    status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
    if status.get('status') != 'completed' or status.get('smoke_test'):
        raise ValueError('Training must finish on the full dataset before deployment.')
    detector = Detector.load(run / 'weights/best.pt', 'cpu')
    if detector.arch != 'yolo11n' or detector.mode != 'test' or detector.size != 640:
        raise ValueError('Expected YOLO11n, mode=test (single image), imgsz=640.')
    torch.set_num_threads(2)
    model = detector.model.eval()
    head = model.model[-1]
    head.export, head.format, head.dynamic = True, 'onnx', False
    destination = run / 'deploy'
    destination.mkdir(exist_ok=True)
    output = destination / 'pcb_yolo11n.onnx'
    example = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        model(example)
        torch.onnx.export(model, example, str(output), input_names=['images'],
                          output_names=['predictions'], opset_version=17,
                          dynamo=False, do_constant_folding=True)
    network = ort.InferenceSession(str(output), providers=['CPUExecutionProvider'])
    if network.get_inputs()[0].shape != [1, 3, 640, 640]:
        raise ValueError('Unexpected ONNX input shape.')
    # Compare raw outputs on real held-out images; do not change model selection.
    checked = []
    for row in manifest('val', 5):
        template, tested = load_pair(project_path(row['template']), project_path(row['tested']))
        x = resize_input(template, tested, 'test', 640)
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()[None] / 255.
        with torch.no_grad():
            expected = model(tensor).numpy()
        actual = network.run(None, {'images': tensor.numpy()})[0]
        np.testing.assert_allclose(actual, expected, atol=.02, rtol=.001)
        checked.append(row['id'])
    metadata = dict(model='yolo11n', mode='test', imgsz=640, run=args.run,
                    status=status, config=detector.config, verified_validation_ids=checked)
    (destination / 'model_info.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Exported and verified: {output} ({output.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
