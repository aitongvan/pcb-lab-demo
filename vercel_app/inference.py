"""ONNX inference matching the trained project's preprocessing and NMS."""
from functools import lru_cache
from pathlib import Path
import io
import time
import warnings

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
VI = ['Đứt mạch', 'Chập mạch', 'Khuyết cạnh', 'Gai thừa', 'Đồng thừa', 'Lỗ kim']
COLORS = ['#f87171', '#fbbf24', '#60a5fa', '#a78bfa', '#34d399', '#fb923c']
SIZE = 640
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 24_000_000
MODELS = {'yolov8n': ('YOLOv8n', 'pcb.onnx'),
          'yolo11n': ('YOLO11n', 'pcb_yolo11n.onnx')}


def available_models():
    return [dict(id=key, name=name, available=(ROOT / 'model' / filename).is_file())
            for key, (name, filename) in MODELS.items()]


def read_image(content):
    if not content or len(content) > MAX_IMAGE_BYTES:
        raise ValueError('Chọn ảnh có dung lượng tối đa 4 MB.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError('Chọn ảnh có tối đa 24 triệu điểm ảnh.')
                return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        raise ValueError('Không đọc được ảnh. Hãy dùng PNG, JPEG, BMP hoặc TIFF hợp lệ.') from error


@lru_cache(maxsize=2)
def session(model='yolov8n'):
    if model not in MODELS:
        raise ValueError('Mô hình không hợp lệ.')
    path = ROOT / 'model' / MODELS[model][1]
    if not path.is_file():
        raise ValueError('YOLO11n một ảnh chưa được triển khai. Cần huấn luyện và bổ sung trọng số.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options,
                               providers=['CPUExecutionProvider'])


def nms(boxes, scores, classes, limit=300):
    order = np.argsort(-scores, kind='stable')
    kept = []
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    while order.size and (limit is None or len(kept) < limit):
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        top_left = np.maximum(boxes[current, :2], boxes[remaining, :2])
        bottom_right = np.minimum(boxes[current, 2:], boxes[remaining, 2:])
        intersection = np.maximum(bottom_right - top_left, 0).prod(axis=1)
        union = area[current] + area[remaining] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = remaining[(classes[remaining] != classes[current]) | (iou <= .5)]
    return kept


def predict_tile(gray, confidence, model='yolov8n'):
    height, width = gray.shape
    resized = np.array(Image.fromarray(gray).resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    tensor = np.repeat(resized[None, None], 3, axis=1).astype(np.float32) / 255.
    network = session(model)
    raw = network.run(None, {network.get_inputs()[0].name: tensor})[0][0].T
    classes = raw[:, 4:].argmax(axis=1)
    scores = raw[:, 4:].max(axis=1)
    mask = (scores >= confidence) & np.isfinite(raw).all(axis=1)
    centers = raw[mask, :4]
    scores, classes = scores[mask], classes[mask]
    boxes = np.concatenate((centers[:, :2] - centers[:, 2:] / 2,
                            centers[:, :2] + centers[:, 2:] / 2), axis=1)
    kept = nms(boxes, scores, classes)
    boxes = boxes[kept].astype(np.float64) * np.array([width / SIZE, height / SIZE] * 2)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    detections = []
    for box, score, label in zip(boxes, scores[kept], classes[kept]):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = int(label)
        detections.append(dict(class_id=label, class_name=NAMES[label], class_vi=VI[label],
                               score=float(score), bbox_xyxy=box.tolist()))
    return detections


def predict(image, confidence=.25, enhance=False, model='yolov8n'):
    if not .05 <= confidence <= .95:
        raise ValueError('Ngưỡng tin cậy phải từ 0,05 đến 0,95.')
    gray = image.convert('L')
    if enhance:
        gray = ImageOps.autocontrast(gray, cutoff=1)
    pixels = np.asarray(gray)
    height, width = pixels.shape
    session(model)  # Model initialization is excluded from the processing measurement.
    start = time.perf_counter()
    if max(width, height) <= 2 * SIZE:
        detections = predict_tile(pixels, confidence, model)
        count = 1
    else:
        def starts(length):
            last = max(0, length - SIZE)
            return [*range(0, last, SIZE * 4 // 5), last]

        xs, ys = starts(width), starts(height)
        count = len(xs) * len(ys)
        detections = []
        for y in ys:
            for x in xs:
                local = predict_tile(pixels[y:y + SIZE, x:x + SIZE], confidence, model)
                for detection in local:
                    box = detection['bbox_xyxy']
                    detection['bbox_xyxy'] = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
                detections.extend(local)
        if detections:
            kept = nms(np.array([d['bbox_xyxy'] for d in detections]),
                       np.array([d['score'] for d in detections]),
                       np.array([d['class_id'] for d in detections]), limit=None)
            detections = [detections[index] for index in kept]
    return dict(predictions=detections, processing_ms=(time.perf_counter() - start) * 1000,
                tiles_scanned=count, width=width, height=height, confidence=confidence,
                contrast_enhanced=enhance, model=MODELS[model][0], runtime='ONNX Runtime CPU')


def annotate(image, predictions, selected=-1):
    result = image.copy()
    draw = ImageDraw.Draw(result)
    for index, prediction in enumerate(predictions):
        box = prediction['bbox_xyxy']
        color = COLORS[prediction['class_id']]
        draw.rectangle(box, outline=color, width=3)
        text = f"{prediction['class_name']} {prediction['score']:.2f}"
        position = (box[0], max(0, box[1] - 16))
        draw.rectangle(draw.textbbox(position, text), fill='black')
        draw.text(position, text, fill=color)
        if index == selected:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            radius = max(15, np.hypot(box[2] - box[0], box[3] - box[1]) / 2 + 5)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         outline='#b8ff65', width=4)
    return result
