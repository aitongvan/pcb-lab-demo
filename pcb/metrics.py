"""Shared detection evaluator: score-sorted matching, 101-point interpolated AP.

All models use the same implementation. This is not DeepPCB's legacy IoU>0.33 evaluator.
"""
import numpy as np
from .common import NAMES

def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0)
    intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
    area = np.maximum(0, np.asarray(box[2:]) - box[:2]).prod()
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(area + areas - intersection, 1e-12)

def _match(preds, ground_truth, threshold):
    matched = {key: set() for key in ground_truth}
    correct = []
    for image_id, det in preds:
        boxes = ground_truth.get(image_id, [])
        overlaps = iou(np.asarray(det['bbox_xyxy']), boxes)
        candidates = [j for j in range(len(boxes)) if j not in matched[image_id] and overlaps[j] >= threshold]
        found = bool(candidates)
        if found:
            matched[image_id].add(max(candidates, key=lambda j: overlaps[j]))
        correct.append(found)
    return np.array(correct, dtype=bool)

def evaluate(records, confidence=0.25, iou_threshold=0.5):
    thresholds = np.arange(0.5, 0.96, 0.05)
    per_class, all_ap, total_tp, total_fp, total_gt = [], [], 0, 0, 0
    for cls, name in enumerate(NAMES):
        gt = {r['id']: [b for c,b in zip(r['classes'], r['boxes']) if c == cls] for r in records}
        ngt = sum(map(len, gt.values()))
        preds = sorted([(r['id'], det) for r in records for det in r['predictions'] if det['class_id'] == cls],
                       key=lambda pair: pair[1]['score'], reverse=True)
        aps = []
        for threshold in thresholds:
            tp = _match(preds, gt, threshold)
            cum_tp, cum_fp = np.cumsum(tp), np.cumsum(~tp)
            recall = cum_tp / max(ngt, 1)
            precision = cum_tp / np.maximum(cum_tp + cum_fp, 1)
            aps.append(float(np.mean([precision[recall >= r].max(initial=0) for r in np.linspace(0,1,101)])) if ngt else None)
        selected = [p for p in preds if p[1]['score'] >= confidence]
        tp = int(_match(selected, gt, iou_threshold).sum())
        fp, fn = len(selected)-tp, ngt-tp
        precision, recall = tp/max(tp+fp,1), tp/max(ngt,1)
        per_class.append(dict(name=name, ground_truth=ngt, tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
                              f1=2*precision*recall/max(precision+recall,1e-12), ap50=aps[0],
                              ap50_95=float(np.mean(aps)) if ngt else None))
        if ngt: all_ap.append(aps)
        total_tp += tp; total_fp += fp; total_gt += ngt
    p, r = total_tp/max(total_tp+total_fp,1), total_tp/max(total_gt,1)
    return dict(images=len(records), ground_truth=total_gt, tp=total_tp, fp=total_fp, fn=total_gt-total_tp,
                precision=p, recall=r, f1=2*p*r/max(p+r,1e-12),
                map50=float(np.mean([a[0] for a in all_ap])) if all_ap else 0.,
                map50_95=float(np.mean(all_ap)) if all_ap else 0., confidence=confidence,
                match_iou=iou_threshold, evaluator='score-sorted class-aware one-to-one; 101 recall samples; absent GT classes excluded from mAP',
                per_class=per_class)
