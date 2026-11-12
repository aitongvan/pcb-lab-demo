import csv
import hashlib
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .common import ROOT, NAMES, VI, COLORS, save_json, read_json, project_path

def read_boxes(path, width, height):
    boxes, classes = [], []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = re.split(r'[,\s]+', line.strip())
        if len(fields) != 5:
            raise ValueError(f'{path}:{number}: cần đúng 5 giá trị')
        x1, y1, x2, y2, label = map(float, fields)
        if not (all(np.isfinite([x1, y1, x2, y2, label])) and
                0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and
                label.is_integer() and 1 <= label <= 6):
            raise ValueError(f'{path}:{number}: hộp hoặc mã lớp không hợp lệ: {line}')
        boxes.append([x1, y1, x2, y2])
        classes.append(int(label) - 1)
    return np.array(boxes, dtype=np.float32).reshape(-1, 4), np.array(classes, dtype=np.int64)

def load_pair(template, tested):
    with Image.open(template) as im:
        t = np.array(im.convert('L'))
    with Image.open(tested) as im:
        i = np.array(im.convert('L'))
    if t.shape != i.shape:
        raise ValueError(f'Hai ảnh phải cùng kích thước: {t.shape} và {i.shape}')
    if min(i.shape) < 16:
        raise ValueError('Ảnh quá nhỏ để xử lý')
    return t, i

def yolo_rows(boxes, classes, width, height):
    result=[]
    for (x1,y1,x2,y2),cls in zip(boxes,classes):
        result.append(f'{int(cls)} {(x1+x2)/(2*width):.8f} {(y1+y2)/(2*height):.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}')
    return result

def represent(t, i, mode):
    if mode == 'pair':
        return np.stack([t, i], axis=-1)
    gray = i if mode == 'test' else np.abs(i.astype(np.int16) - t.astype(np.int16)).astype(np.uint8)
    if mode not in ('test', 'diff'):
        raise ValueError(f'Dạng đầu vào không hợp lệ: {mode}')
    return np.repeat(gray[..., None], 3, axis=-1)

def resize_input(t, i, mode, size):
    t = np.array(Image.fromarray(t).resize((size, size), Image.Resampling.BILINEAR))
    i = np.array(Image.fromarray(i).resize((size, size), Image.Resampling.BILINEAR))
    return represent(t, i, mode)

def annotate(image, detections):
    image = image.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    for det in detections:
        c = int(det['class_id'])
        box = det['bbox_xyxy']
        draw.rectangle(box, outline=COLORS[c], width=3)
        # ASCII labels work on all Windows fonts; Vietnamese appears in the UI table.
        text = f'{NAMES[c]} {det["score"]:.2f}'
        x, y = box[0], max(0, box[1] - 16)
        bounds = draw.textbbox((x, y), text)
        draw.rectangle(bounds, fill='black')
        draw.text((x, y), text, fill=COLORS[c])
    return image

def prepare(raw, val_fraction=0.2, seed=42):
    raw = Path(raw).resolve()
    if (raw / 'PCBData').is_dir():
        raw = raw / 'PCBData'
    out = ROOT / 'data/processed'
    if not 0 < val_fraction < 1:
        raise ValueError('Tỷ lệ Validation phải nằm giữa 0 và 1')
    pools = {}
    for split, filename in [('trainval', 'trainval.txt'), ('test', 'test.txt')]:
        listing = raw / filename
        if not listing.exists():
            raise FileNotFoundError(f'Không thấy {listing}. Chạy scripts/download_data.ps1 trước.')
        rows = []
        for line in listing.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            parts = line.split()
            base, label = raw / parts[0], raw / parts[1]
            tested = base.with_name(base.stem + '_test' + base.suffix)
            template = base.with_name(base.stem + '_temp' + base.suffix)
            if not tested.exists() and base.exists():
                tested = base
            t, image = load_pair(template, tested)
            height, width = image.shape
            boxes, classes = read_boxes(label, width, height)
            rows.append(dict(id=base.stem, template=str(template.relative_to(ROOT)),
                             tested=str(tested.relative_to(ROOT)), label=str(label.relative_to(ROOT)),
                             width=width, height=height, boxes=boxes.tolist(), classes=classes.tolist(),
                             group=Path(parts[0]).parts[0], sha256=hashlib.sha256(image.tobytes()).hexdigest()))
        pools[split] = rows
    # Preserve the official test split. Sample-level holdout matches the report;
    # this is NOT a board-independent split. Audit duplicate tested images explicitly.
    trainval = pools['trainval'].copy()
    random.Random(seed).shuffle(trainval)
    nval = max(1, round(len(trainval) * val_fraction))
    rows = []
    for split, samples in [('val', trainval[:nval]), ('train', trainval[nval:]), ('test', pools['test'])]:
        for row in samples:
            rows.append(dict(row, split=split))
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Mã cặp trùng trong các danh sách gốc')
    hashes = defaultdict(set)
    for row in rows:
        hashes[row['sha256']].add(row['split'])
    overlap = [h for h, splits in hashes.items() if len(splits) > 1]
    summary = {'seed': seed, 'split_policy': 'official test; random pair holdout within official trainval',
               'board_independent': False, 'coordinate_policy': 'source xyxy kept as continuous edges; no implicit +/-1',
               'duplicate_test_images_across_splits': overlap, 'splits': {}}
    for split in ['train', 'val', 'test']:
        group = [r for r in rows if r['split'] == split]
        count = Counter(c for r in group for c in r['classes'])
        summary['splits'][split] = {'pairs': len(group), 'boxes': sum(count.values()),
                                    'per_class': {NAMES[c]: count[c] for c in range(6)}}
    save_json(out / 'manifest.json', rows)
    save_json(out / 'audit.json', summary)
    save_json(out / 'splits.json', {s: [r['id'] for r in rows if r['split'] == s] for s in ['train','val','test']})
    save_json(out / 'class_mapping.json', [{'source': c+1, 'yolo': c, 'fasterrcnn': c+1, 'name': NAMES[c], 'vi': VI[c]} for c in range(6)])
    for row in rows:
        target=out/'yolo_labels'/row['split']/f'{row["id"]}.txt'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('\n'.join(yolo_rows(row['boxes'],row['classes'],row['width'],row['height']))+'\n',encoding='utf-8')
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['id','split','group','template','tested','label','width','height'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})
    previews = out / 'previews'
    previews.mkdir(parents=True, exist_ok=True)
    chosen, seen = [], set()
    for row in rows:
        if set(row['classes']) - seen:
            chosen.append(row); seen.update(row['classes'])
        if len(seen) == 6:
            break
    for row in chosen:
        with Image.open(project_path(row['tested'])) as im:
            dets = [{'class_id': c, 'score': 1., 'bbox_xyxy': b} for c,b in zip(row['classes'], row['boxes'])]
            annotate(im, dets).save(previews / f'{row["id"]}_labels.png')
    if overlap:
        raise ValueError('Có ảnh kiểm tra trùng nội dung giữa các tập. Xem audit.json; xử lý trước khi huấn luyện.')
    return summary

def manifest(split=None, limit=None):
    path = ROOT / 'data/processed/manifest.json'
    if not path.exists():
        raise FileNotFoundError('Chưa chuẩn bị dữ liệu. Chạy: python -m pcb prepare')
    rows = read_json(path)
    if split:
        rows = [r for r in rows if r['split'] == split]
    return rows[:limit] if limit else rows

