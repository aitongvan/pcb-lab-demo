"""Validate full training artifacts before installing YOLO11n in the web app."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path, help='Extracted completed run directory')
    parser.add_argument('--reference-splits', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    config = read(run / 'config.json')
    status = read(run / 'status.json')
    assert config['model'] == 'yolo11n' and config['mode'] == 'test'
    assert config['imgsz'] == 640 and not config['smoke_test']
    assert status['status'] == 'completed' and not status['smoke_test']
    reference, ids = read(args.reference_splits), read(run / 'sample_ids.json')
    for split in ('train', 'val'):
        assert sorted(reference[split]) == sorted(ids[split]), f'{split} IDs differ from baseline'
    metrics = {}
    for split, images, boxes in [('train', 800, 5485), ('val', 200, 1388), ('test', 500, 3140)]:
        report = read(run / f'evaluation_{split}/metrics.json')
        assert report['images'] == images and report['ground_truth'] == boxes
        assert report['split'] == split and not report['limited_evaluation']
        assert not report['checkpoint_smoke_test']
        assert report['confidence'] == .25 and report['match_iou'] == .5
        metrics[split] = report
    info = read(run / 'deploy/model_info.json')
    assert info['model'] == 'yolo11n' and info['mode'] == 'test'
    assert len(info['verified_validation_ids']) == 5
    import onnxruntime as ort
    source = run / 'deploy/pcb_yolo11n.onnx'
    network = ort.InferenceSession(str(source), providers=['CPUExecutionProvider'])
    assert network.get_inputs()[0].shape == [1, 3, 640, 640]
    assert network.get_outputs()[0].shape[1] == 10
    destination = PROJECT / 'vercel_app/model/pcb_yolo11n.onnx'
    assert not destination.exists(), 'Model exists; inspect before replacing it'
    shutil.copy2(source, destination)
    installed = dict(info, sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     training_environment=read(run / 'environment.json'))
    destination.with_suffix('.json').write_text(json.dumps(installed, ensure_ascii=False, indent=2), encoding='utf-8')
    path = PROJECT / 'vercel_app/public/experiments.json'
    experiments = [item for item in read(path) if item['run'] != run.name]
    experiments.append(dict(run=run.name, model='yolo11n', mode='test', metrics=metrics,
                            environment=installed['training_environment']))
    path.write_text(json.dumps(experiments, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Installed real YOLO11n single-image model and verified full evaluation reports.')
    print('Next: verify_model.py --model yolo11n --checkpoint <run>/weights/best.pt')


if __name__ == '__main__':
    main()
