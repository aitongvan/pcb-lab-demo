import argparse
from pathlib import Path
from .common import ROOT, save_json, project_path

def parser():
    p=argparse.ArgumentParser(description='Project học máy phát hiện lỗi PCB')
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare',help='Kiểm tra DeepPCB và tạo danh sách Train/Val/Test')
    q.add_argument('--raw',default=str(ROOT/'data/raw/DeepPCB'))
    q.add_argument('--val-fraction',type=float,default=.2); q.add_argument('--seed',type=int,default=42)
    q=sub.add_parser('train',help='Huấn luyện YOLO/Faster R-CNN')
    q.add_argument('--model',choices=['yolov8n','yolo11n','fasterrcnn'],default='yolov8n')
    q.add_argument('--mode',choices=['test','diff','pair'],default='test')
    q.add_argument('--name',default='yolov8n_test')
    q.add_argument('--epochs',type=int,default=100); q.add_argument('--batch',type=int,default=8)
    q.add_argument('--imgsz',type=int,default=640); q.add_argument('--lr',type=float,default=.001)
    q.add_argument('--patience',type=int,default=20); q.add_argument('--seed',type=int,default=42)
    q.add_argument('--device',default='auto'); q.add_argument('--threads',type=int,default=4)
    q.add_argument('--scratch',action='store_true',help='Không tải trọng số học trước; dùng cho smoke test')
    q.add_argument('--limit-train',type=int); q.add_argument('--limit-val',type=int)
    q.add_argument('--confidence',type=float,default=.25)
    q=sub.add_parser('evaluate',help='Đánh giá checkpoint trên Train/Val/Test')
    q.add_argument('--checkpoint',required=True); q.add_argument('--split',choices=['train','val','test'],default='test')
    q.add_argument('--limit',type=int); q.add_argument('--confidence',type=float,default=.25)
    q.add_argument('--device',default='auto'); q.add_argument('--threads',type=int,default=4)
    q=sub.add_parser('predict',help='Dự đoán một cặp ảnh và lưu kết quả')
    q.add_argument('--checkpoint',required=True); q.add_argument('--template',required=True); q.add_argument('--tested',required=True)
    q.add_argument('--output',default='outputs/prediction'); q.add_argument('--confidence',type=float,default=.25)
    q.add_argument('--device',default='auto')
    return p

def main():
    import sys
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr,'reconfigure'): sys.stderr.reconfigure(encoding='utf-8')
    args=parser().parse_args()
    if args.command=='prepare':
        from .data import prepare
        import json
        print(json.dumps(prepare(args.raw,args.val_fraction,args.seed),ensure_ascii=False,indent=2))
    elif args.command=='train':
        from .engine import train
        train(args)
    elif args.command=='evaluate':
        import torch
        from .engine import Detector,predict_rows
        from .data import manifest
        torch.set_num_threads(args.threads)
        detector=Detector.load(project_path(args.checkpoint),args.device)
        output=Path(project_path(args.checkpoint)).parents[1]/f'evaluation_{args.split}'
        if args.limit: output=output.with_name(output.name+f'_limit{args.limit}')
        metrics=predict_rows(detector,manifest(args.split,args.limit),args.confidence,output)
        metrics['split']=args.split; metrics['limited_evaluation']=bool(args.limit)
        metrics['checkpoint_smoke_test']=detector.config.get('smoke_test',False)
        save_json(output/'metrics.json',metrics)
        print(f'mAP50={metrics["map50"]:.4f}; mAP50-95={metrics["map50_95"]:.4f}; kết quả: {output}')
    elif args.command=='predict':
        from PIL import Image
        from .engine import Detector
        from .data import load_pair,annotate
        detector=Detector.load(project_path(args.checkpoint),args.device)
        t,i=load_pair(args.template,args.tested)
        detections,ms=detector.predict_arrays(t,i,args.confidence)
        output=project_path(args.output); output.mkdir(parents=True,exist_ok=True)
        annotate(Image.fromarray(i),detections).save(output/'result.png')
        save_json(output/'prediction.json',dict(detections=detections,processing_ms=ms,config=detector.config))
        print(f'{len(detections)} dự đoán; đã lưu {output}')

if __name__=='__main__':
    main()

