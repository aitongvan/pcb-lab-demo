import csv
import os
import platform
import random
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from .common import ROOT, NAMES, VI, save_json, project_path
from .data import load_pair, resize_input, manifest, annotate
from .metrics import evaluate

os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'outputs/.ultralytics'))
os.environ.setdefault('YOLO_AUTOINSTALL','false')
os.environ.setdefault('TORCH_HOME',str(ROOT/'data/cache/torch'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/.matplotlib'))

def device_name(request='auto'):
    if request == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if request.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA chưa sẵn sàng. Dùng --device cpu hoặc cài PyTorch CUDA phù hợp.')
    return request

class PairDataset(Dataset):
    def __init__(self, rows, mode, size, augment=False):
        self.rows, self.mode, self.size, self.augment = rows, mode, size, augment

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        t, i = load_pair(project_path(row['template']), project_path(row['tested']))
        x = resize_input(t, i, self.mode, self.size)
        boxes = np.asarray(row['boxes'], dtype=np.float32).reshape(-1, 4).copy()
        boxes *= np.array([self.size/row['width'],self.size/row['height']]*2)
        if self.augment and random.random() < .5:
            x = np.flip(x, axis=1)
            boxes[:, [0,2]] = self.size-boxes[:,[2,0]]
        if self.augment and random.random() < .5:
            x = np.flip(x, axis=0)
            boxes[:, [1,3]] = self.size-boxes[:,[3,1]]
        image = torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float()/255.
        return image, torch.tensor(boxes), torch.tensor(row['classes'],dtype=torch.int64)

def collate(batch): return list(zip(*batch))

class Detector:
    def __init__(self, config, device='auto', pretrained=False):
        self.config = dict(config)
        self.device = torch.device(device_name(device))
        self.size = int(config['imgsz'])
        self.mode, self.arch = config['mode'], config['model']
        if self.size < 64 or self.size % 32:
            raise ValueError('imgsz phải >=64 và chia hết cho 32')
        if self.arch == 'fasterrcnn':
            if self.mode == 'pair':
                raise ValueError('Faster R-CNN đối chứng chỉ hỗ trợ test/diff trong project này')
            from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
            from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
            self.model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
                weights_backbone=None, min_size=self.size, max_size=self.size,
                box_score_thresh=.001, box_nms_thresh=.5, box_detections_per_img=300)
            features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(features, 7)
        else:
            if self.arch not in ('yolov8n','yolo11n'):
                raise ValueError('Chỉ hỗ trợ yolov8n, yolo11n, fasterrcnn')
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.cfg import get_cfg
            self.model = DetectionModel(f'{self.arch}.yaml', ch=2 if self.mode=='pair' else 3, nc=6, verbose=False)
            self.model.args = get_cfg()
            if pretrained:
                from ultralytics import YOLO
                pretrained_dir=ROOT/'models/pretrained'
                pretrained_dir.mkdir(parents=True,exist_ok=True)
                source = YOLO(str(pretrained_dir/f'{self.arch}.pt')).model.state_dict()
                target = self.model.state_dict()
                first_key = 'model.0.conv.weight'
                if self.mode == 'pair' and first_key in source:
                    source[first_key] = source[first_key].sum(dim=1,keepdim=True).repeat(1,2,1,1)/2
                matching = {k:v for k,v in source.items() if k in target and v.shape == target[k].shape}
                self.model.load_state_dict(matching,strict=False)
                self.config['pretrained_tensors_loaded'] = len(matching)
        self.model.to(self.device)

    def loss(self, images, boxes, classes):
        self.model.train()
        images = [im.to(self.device) for im in images]
        if self.arch == 'fasterrcnn':
            # Batch size 1 at small resolution: pretrained backbone uses frozen BN by default;
            # for randomly initialized models keep BN statistics stable.
            if len(images) == 1:
                for module in self.model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d): module.eval()
            targets = [dict(boxes=b.to(self.device), labels=(c+1).to(self.device)) for b,c in zip(boxes,classes)]
            return sum(self.model(images, targets).values())
        xywh, batch_idx, labels = [], [], []
        for idx,(b,c) in enumerate(zip(boxes,classes)):
            converted = b.clone()
            converted[:,:2] = (b[:,:2]+b[:,2:])/2
            converted[:,2:] = b[:,2:]-b[:,:2]
            xywh.append(converted/self.size)
            labels.append(c[:,None].float())
            batch_idx.append(torch.full((len(c),),idx,dtype=torch.float32))
        batch = dict(img=torch.stack(images), bboxes=torch.cat(xywh).to(self.device),
                     cls=torch.cat(labels).to(self.device), batch_idx=torch.cat(batch_idx).to(self.device))
        loss, _ = self.model(batch)
        return loss.sum()

    @torch.inference_mode()
    def predict_arrays(self, template, tested, confidence=.25, nms_iou=.5):
        if not 0 <= confidence <= 1 or not 0 < nms_iou <= 1:
            raise ValueError('Ngưỡng điểm hoặc IoU không hợp lệ')
        if template.shape != tested.shape:
            raise ValueError('Ảnh mẫu và ảnh kiểm tra phải cùng kích thước')
        if self.device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter()
        self.model.eval()
        x = resize_input(template, tested, self.mode, self.size)
        tensor = torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).float().to(self.device)/255.
        if self.arch == 'fasterrcnn':
            self.model.roi_heads.nms_thresh = nms_iou
            pred = self.model([tensor])[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            classes = pred['labels'].cpu().numpy()-1
        else:
            try:
                from ultralytics.utils.nms import non_max_suppression
            except ImportError:
                from ultralytics.utils.ops import non_max_suppression
            raw = self.model(tensor[None])
            predictions = non_max_suppression(raw,conf_thres=confidence,iou_thres=nms_iou,nc=6,max_det=300)[0]
            values = predictions.cpu().numpy()
            boxes, scores, classes = values[:,:4], values[:,4], values[:,5].astype(int)
        h,w = tested.shape
        detections = []
        for box,score,cls in zip(boxes,scores,classes):
            cls = int(cls)
            if score < confidence or cls not in range(6): continue
            box = box*np.array([w/self.size,h/self.size]*2)
            box[[0,2]] = np.clip(box[[0,2]],0,w)
            box[[1,3]] = np.clip(box[[1,3]],0,h)
            if not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]: continue
            detections.append(dict(class_id=cls,class_name=NAMES[cls],class_vi=VI[cls],score=float(score),bbox_xyxy=box.tolist()))
        if self.device.type == 'cuda': torch.cuda.synchronize()
        elapsed = (time.perf_counter()-start)*1000
        return detections, elapsed

    def predict_image(self, image, confidence=.25):
        """Detect from one grayscale image, tiling boards larger than twice the input size."""
        if self.mode != 'test':
            raise ValueError('Checkpoint phải là mô hình dùng một ảnh kiểm tra.')
        height, width = image.shape
        if max(height, width) <= 2 * self.size:
            detections, elapsed = self.predict_arrays(image, image, confidence)
            return detections, elapsed, 1

        from torchvision.ops import batched_nms

        started = time.perf_counter()
        size = self.size
        stride = size * 4 // 5

        def tile_starts(length):
            last = max(0, length - size)
            return [*range(0, last, stride), last]

        detections = []
        xs = tile_starts(width)
        ys = tile_starts(height)
        for y in ys:
            for x in xs:
                tile = image[y:y + size, x:x + size]
                local, _ = self.predict_arrays(tile, tile, confidence)
                for detection in local:
                    x1, y1, x2, y2 = detection['bbox_xyxy']
                    detection['bbox_xyxy'] = [x1 + x, y1 + y, x2 + x, y2 + y]
                detections.extend(local)

        if detections:
            boxes = torch.tensor([d['bbox_xyxy'] for d in detections], dtype=torch.float32)
            scores = torch.tensor([d['score'] for d in detections], dtype=torch.float32)
            classes = torch.tensor([d['class_id'] for d in detections], dtype=torch.int64)
            kept = batched_nms(boxes, scores, classes, iou_threshold=.5)
            detections = [detections[index] for index in kept.tolist()]

        elapsed = (time.perf_counter() - started) * 1000
        return detections, elapsed, len(xs) * len(ys)

    def save(self,path,**metadata):
        path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(format='pcb-project-v1',config=self.config,names=NAMES,
                        state_dict={k:v.detach().cpu() for k,v in self.model.state_dict().items()},**metadata),path)

    @classmethod
    def load(cls,path,device='auto'):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('format') != 'pcb-project-v1':
            raise ValueError('Chọn checkpoint do project tạo (outputs/<lần chạy>/weights/best.pt).')
        detector=cls(checkpoint['config'],device,False)
        detector.model.load_state_dict(checkpoint['state_dict'])
        detector.metadata={k:v for k,v in checkpoint.items() if k!='state_dict'}
        return detector

def predict_rows(detector,rows,confidence=.25,output=None,preview_count=12):
    records,times=[],[]
    if rows:
        t,i=load_pair(project_path(rows[0]['template']),project_path(rows[0]['tested']))
        detector.predict_arrays(t,i,.001)  # warm up, excluded from timing
    for index,row in enumerate(rows):
        t,i=load_pair(project_path(row['template']),project_path(row['tested']))
        predictions,ms=detector.predict_arrays(t,i,.001)
        records.append(dict(id=row['id'],classes=row['classes'],boxes=row['boxes'],predictions=predictions))
        times.append(ms)
        if output and index<preview_count:
            destination=Path(output)/'previews'; destination.mkdir(parents=True,exist_ok=True)
            annotate(Image.fromarray(i),[p for p in predictions if p['score']>=confidence]).save(destination/f'{row["id"]}.png')
    metrics=evaluate(records,confidence=confidence)
    metrics['mean_preprocess_inference_postprocess_ms']=float(np.mean(times)) if times else 0.
    metrics['timing_scope']='arrays already loaded; resize + tensor conversion + network + postprocessing; warmup excluded; file IO excluded'
    if output:
        save_json(Path(output)/'metrics.json',metrics)
        save_json(Path(output)/'predictions.json',records)
        with (Path(output)/'per_class.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(metrics['per_class'][0])); writer.writeheader(); writer.writerows(metrics['per_class'])
    return metrics

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.epochs<1 or args.batch<1: raise ValueError('epochs và batch phải dương')
    rows=manifest('train',args.limit_train); val=manifest('val',args.limit_val)
    if not rows or not val: raise ValueError('Train và Validation không được rỗng')
    audit_path=ROOT/'data/processed/audit.json'
    from .common import read_json
    if read_json(audit_path)['duplicate_test_images_across_splits']:
        raise ValueError('Dữ liệu trùng giữa các tập. Xem audit.json.')
    config=dict(model=args.model,mode=args.mode,imgsz=args.imgsz,seed=args.seed,
                pretrained=not args.scratch,smoke_test=bool(args.limit_train or args.limit_val),
                training_loop='project PyTorch loop; AdamW; synchronized flips; no stock Ultralytics trainer/EMA',
                learning_rate=args.lr,batch=args.batch,epochs=args.epochs,confidence=args.confidence)
    run=ROOT/'outputs'/args.name
    if run.exists(): raise FileExistsError(f'{run} đã tồn tại. Chọn --name mới để không ghi đè.')
    run.mkdir(parents=True)
    save_json(run/'config.json',config)
    import importlib.metadata as md
    save_json(run/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),
                 packages={m:md.version(m) for m in ['torch','torchvision','ultralytics','numpy']},
                 device=device_name(args.device)))
    save_json(run/'sample_ids.json',dict(train=[r['id'] for r in rows],val=[r['id'] for r in val]))
    detector=Detector(config,args.device,not args.scratch)
    optimizer=torch.optim.AdamW(detector.model.parameters(),lr=args.lr,weight_decay=.0005)
    loader=DataLoader(PairDataset(rows,args.mode,args.imgsz,True),batch_size=args.batch,shuffle=True,num_workers=0,collate_fn=collate)
    best,stale,history=-1.,0,[]
    try:
        for epoch in range(1,args.epochs+1):
            started=time.perf_counter(); total=0.
            for images,boxes,classes in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=detector.loss(images,boxes,classes)
                if not torch.isfinite(loss): raise RuntimeError('Mất mát không hữu hạn. Kiểm tra dữ liệu/cấu hình.')
                loss.backward(); torch.nn.utils.clip_grad_norm_(detector.model.parameters(),10.)
                optimizer.step(); total+=float(loss.detach())
            metrics=predict_rows(detector,val,args.confidence)
            record=dict(epoch=epoch,train_loss=total/max(len(loader),1),map50=metrics['map50'],
                        map50_95=metrics['map50_95'],precision=metrics['precision'],recall=metrics['recall'],
                        seconds=time.perf_counter()-started)
            history.append(record)
            with (run/'history.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
            detector.save(run/'weights/last.pt',epoch=epoch,validation=metrics)
            if metrics['map50_95']>best:
                best=metrics['map50_95']; stale=0
                detector.save(run/'weights/best.pt',epoch=epoch,validation=metrics)
                save_json(run/'validation.json',metrics)
            else: stale+=1
            print(f'Epoch {epoch}/{args.epochs}: loss={record["train_loss"]:.4f}; val mAP50={metrics["map50"]:.4f}; mAP50-95={metrics["map50_95"]:.4f}',flush=True)
            if stale>=args.patience: break
    except BaseException as error:
        save_json(run/'status.json',dict(status='interrupted_or_failed',message=str(error)))
        raise
    save_json(run/'status.json',dict(status='completed',epochs=len(history),best_map50_95=best,
                                    smoke_test=config['smoke_test']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['epoch'] for r in history],[r['train_loss'] for r in history]); axes[0].set_title('Train loss')
    axes[1].plot([r['epoch'] for r in history],[r['map50_95'] for r in history]); axes[1].set_title('Validation mAP50-95')
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(run/'learning_curves.png'); plt.close(fig)
    print(f'Đã lưu: {run}',flush=True)
    return run
