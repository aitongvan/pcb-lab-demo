import tempfile
import unittest
from pathlib import Path
import numpy as np
from pcb.data import read_boxes, represent, load_pair, yolo_rows
from pcb.metrics import evaluate, iou
from PIL import Image

class DetectionTests(unittest.TestCase):
    def det(self,c=0,box=None,score=.9):
        return dict(class_id=c,bbox_xyxy=box or [0,0,10,10],score=score)

    def record(self,preds,classes=None,boxes=None):
        return dict(id='a',classes=[0] if classes is None else classes,
                    boxes=[[0,0,10,10]] if boxes is None else boxes,predictions=preds)

    def test_iou_continuous_coordinates(self):
        np.testing.assert_allclose(iou(np.array([0,0,10,10]),[[0,0,10,10],[5,0,15,10]]),[1,1/3])

    def test_perfect_detection_ap(self):
        result=evaluate([self.record([self.det()])])
        self.assertEqual(result['tp'],1)
        self.assertEqual(result['map50_95'],1.)

    def test_duplicate_prediction_only_one_match(self):
        result=evaluate([self.record([self.det(),self.det(score=.8)])])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))
        self.assertEqual(result['precision'],.5)

    def test_wrong_class_not_true_positive(self):
        result=evaluate([self.record([self.det(c=1)])])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))

    def test_matching_uses_unmatched_alternative(self):
        rows=[self.record([self.det(box=[0,0,10,10]),self.det(box=[1,0,11,10],score=.8)],
                          classes=[0,0],boxes=[[0,0,10,10],[2,0,12,10]])]
        result=evaluate(rows)
        self.assertEqual(result['tp'],2)

    def test_empty_predictions(self):
        result=evaluate([self.record([])])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,0,1))
        self.assertEqual(result['map50'],0.)

    def test_confidence_and_ap_are_separate(self):
        result=evaluate([self.record([self.det(score=.1)])],confidence=.25)
        self.assertEqual(result['tp'],0)
        self.assertEqual(result['map50'],1.)

    def test_class_mapping_and_invalid_box(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.txt'
            path.write_text('0 0 10 10 5\n1,1,9,9,6',encoding='utf-8')
            boxes,classes=read_boxes(path,20,20)
            self.assertEqual(classes.tolist(),[4,5])
            path.write_text('10 0 0 10 1',encoding='utf-8')
            with self.assertRaises(ValueError): read_boxes(path,20,20)

    def test_difference_no_uint8_wraparound(self):
        t=np.array([[255,0]],dtype=np.uint8)
        i=np.array([[0,255]],dtype=np.uint8)
        np.testing.assert_array_equal(represent(t,i,'diff')[...,0],[[255,255]])
        np.testing.assert_array_equal(represent(t,i,'pair')[...,0],t)
        np.testing.assert_array_equal(represent(t,i,'pair')[...,1],i)

    def test_pair_size_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.png',Path(directory)/'b.png'
            Image.new('L',(20,20)).save(a); Image.new('L',(30,20)).save(b)
            with self.assertRaises(ValueError): load_pair(a,b)

    def test_yolo_normalization(self):
        line=yolo_rows([[10,20,30,60]],[5],100,200)[0]
        values=list(map(float,line.split()))
        np.testing.assert_allclose(values,[5,.2,.2,.2,.2])

if __name__=='__main__': unittest.main()

