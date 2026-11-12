"""Integration checks on real Train pairs. Not a scientific experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gc
import torch
from pcb.common import ROOT,project_path,save_json
from pcb.data import manifest,load_pair
from pcb.engine import Detector,PairDataset,collate

def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    torch.set_num_threads(2)
    rows=manifest('train',2)
    checks=[]
    for arch,mode in [('yolov8n','test'),('yolov8n','pair'),('yolo11n','diff'),('fasterrcnn','test')]:
        config=dict(model=arch,mode=mode,imgsz=160,smoke_test=True)
        detector=Detector(config,'cpu',False)
        batch=collate([PairDataset(rows,mode,160)[i] for i in range(2)])
        loss=detector.loss(*batch)
        assert torch.isfinite(loss),f'Non-finite loss: {arch}/{mode}'
        loss.backward()
        grad=next(p.grad for p in detector.model.parameters() if p.grad is not None)
        assert torch.isfinite(grad).all()
        t,i=load_pair(project_path(rows[0]['template']),project_path(rows[0]['tested']))
        preds,ms=detector.predict_arrays(t,i,.01)
        # Verify roundtrip for the custom 2-channel checkpoint, which is especially easy to break.
        if mode=='pair':
            destination=ROOT/'outputs/integration_pair/weights/best.pt'
            detector.save(destination,validation={},epoch=0)
            restored=Detector.load(destination,'cpu')
            again,_=restored.predict_arrays(t,i,.01)
            assert len(again)==len(preds)
            if preds:
                assert abs(again[0]['score']-preds[0]['score'])<1e-6
            save_json(destination.parents[1]/'config.json',config)
            del restored
        checks.append(dict(model=arch,mode=mode,finite_loss=float(loss.detach()),
                           backward='passed',inference='passed',detections_at_001=len(preds),processing_ms=ms))
        print(f'PASS: {arch}/{mode}: loss={float(loss.detach()):.4f}; detections={len(preds)}',flush=True)
        del detector,loss,batch,grad
        gc.collect()
    save_json(ROOT/'outputs/integration_checks.json',checks)

if __name__=='__main__': main()
