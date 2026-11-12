"""Headless Streamlit interaction checks; requires a project checkpoint."""
import io
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from PIL import Image
from pcb.common import ROOT,project_path,save_json

def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    a=AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
    assert not a.exception,[x.message for x in a.exception]
    assert [x.value for x in a.title]==['PCB Lab']
    assert a.sidebar.selectbox[0].value.parents[1].name=='yolov8n_gpu_full_01'
    model_options=a.sidebar.selectbox[0].options
    assert len(model_options)==len(set(model_options))
    assert any('yolov8n_gpu_full_01' in option for option in model_options)
    assert any('kiemtra_gpu_01' in option for option in model_options)
    assert len(a.file_uploader)==1
    assert a.file_uploader[0].label=='Ảnh PCB đầu vào'
    assert not any('hai kênh' in option or 'Ảnh hiệu' in option for option in a.sidebar.selectbox[0].options)
    compare=next(d.value for d in a.dataframe if 'Lượt chạy' in d.value.columns)
    assert len(compare)==4
    assert not compare['Mô hình'].duplicated().any()
    next(r for r in a.radio if r.label=='Nguồn ảnh').set_value('Mẫu DeepPCB').run(timeout=40)
    next(s for s in a.selectbox if s.label=='Tập dữ liệu').set_value('test').run(timeout=40)
    assert not a.exception,[x.message for x in a.exception]
    sample=next(s for s in a.selectbox if s.label=='Chọn ảnh').value
    assert sample['split']=='test'
    assert a.session_state['prediction']['sample_id']==sample['id']
    assert not a.exception,[x.message for x in a.exception]
    assert not a.error,[x.value for x in a.error]
    assert any(m.label=='Vùng lỗi được dự đoán' for m in a.metric)
    assert len(a.get('download_button'))==2
    a.sidebar.slider[0].set_value(.95).run(timeout=40)
    assert not a.exception,[x.message for x in a.exception]
    assert not a.error,[x.value for x in a.error]
    prediction=a.session_state['prediction']
    assert prediction['confidence']==.95
    assert all(item['score']>=.95 for item in prediction['predictions'])
    next(r for r in a.radio if r.label=='Tập đánh giá').set_value('test').run(timeout=40)
    assert a.session_state['prediction'] is prediction
    compare=next(d.value for d in a.dataframe if 'Lượt chạy' in d.value.columns)
    assert all(compare['Cặp ảnh']==500)
    next(r for r in a.radio if r.label=='Nguồn ảnh').set_value('Tải ảnh của bạn').run(timeout=40)
    assert len(a.file_uploader)==1
    assert len(a.get('download_button'))==0
    with Image.open(project_path(sample['tested'])) as image:
        data=io.BytesIO()
        image.crop((0,0,500,500)).save(data,format='PNG')
    a.sidebar.slider[0].set_value(.25)
    a.file_uploader[0].set_value(('anh_pcb.png',data.getvalue(),'image/png')).run(timeout=40)
    assert a.session_state['prediction']['sample_id']=='uploaded'
    assert a.session_state['prediction']['confidence']==.25
    assert not any(b.label=='Phát hiện lỗi' for b in a.button)
    assert not a.exception,[x.message for x in a.exception]
    assert not a.error,[x.value for x in a.error]
    captions=[caption for image in a.image for caption in image.captions if caption]
    assert len(captions)==2 and captions[0]=='500 × 500 px' and captions[1].endswith('Ngưỡng 0.25'),captions
    assert len(a.get('download_button'))==2
    a.file_uploader[0].set_value(('anh_pcb.png',b'invalid image','image/png')).run(timeout=40)
    assert any('Không đọc được ảnh' in e.value for e in a.error)
    assert len(a.get('download_button'))==0
    save_json(ROOT/'outputs/ui_checks.json',dict(initial_render='passed',automatic_prediction='passed',
                 result_downloads='passed',single_image_upload='passed',invalid_image='passed',
                 comparison_full_runs='passed',test_split_filter='passed',confidence_auto_refresh='passed',prediction_reuse='passed'))
    print('PASS: dự đoán và xuất kết quả từ một ảnh tải lên hoặc ảnh kiểm tra DeepPCB.',flush=True)

if __name__=='__main__': main()
