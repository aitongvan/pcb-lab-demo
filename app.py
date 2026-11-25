import base64
import io
import json
from math import hypot
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from pcb.common import ROOT, VI, NAMES, COLORS, read_json, project_path
from pcb.data import manifest, annotate
from pcb.visuals import scan_gif

st.set_page_config(page_title='PCB Lab | Phát hiện lỗi mạch in',page_icon='🔬',layout='wide')
st.markdown(f"<style>{(ROOT / 'assets/interface.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True)

hero_preview = ROOT / 'outputs/yolov8n_gpu_full_01/evaluation_test/previews/20085291.png'
hero_figure = ''
if hero_preview.is_file():
    encoded = base64.b64encode(hero_preview.read_bytes()).decode('ascii')
    hero_figure = (
        '<figure class="pcb-hero-figure"><div class="pcb-scan-frame">'
        f'<img src="data:image/png;base64,{encoded}" '
        'alt="Mẫu DeepPCB 20085291 với các vùng lỗi được mô hình dự đoán">'
        '</div><figcaption>Mẫu DeepPCB 20085291 · Vị trí lỗi dự đoán</figcaption></figure>'
    )
st.markdown(
    '<header class="pcb-hero"><div>'
    '<div class="pcb-eyebrow">Phát hiện lỗi mạch in</div>'
    '<h1>PCB <span class="pcb-lab-word">LAB</span></h1>'
    '<p class="pcb-lead">Phát hiện, phân loại và định vị sáu loại lỗi trên ảnh PCB. '
    'Chọn ảnh mẫu hoặc tải ảnh của bạn để bắt đầu.</p>'
    f'</div>{hero_figure}</header>', unsafe_allow_html=True)

MODES={'test':'Ảnh kiểm tra','diff':'Ảnh hiệu','pair':'Cặp ảnh hai kênh'}
SPLITS={'val':'Validation','test':'Test','train':'Train'}

def run_config(run):
    path=run/'config.json'
    return read_json(path) if path.exists() else {}

def run_label(run):
    cfg=run_config(run)
    suffix=' · Chạy thử' if cfg.get('smoke_test') else ''
    return f'{cfg.get("model",run.name).upper()} · {MODES.get(cfg.get("mode"),"Chưa có cấu hình")}{suffix}'

def highlight_detection(image, box):
    x1, y1, x2, y2 = box
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    scale = max(1, max(image.size) / 600)
    radius = max(15 * scale, hypot(x2 - x1, y2 - y1) / 2 + 5 * scale)
    bounds = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
    highlighted = image.copy()
    draw = ImageDraw.Draw(highlighted)
    draw.ellipse(bounds, outline='#080e18', width=round(7 * scale))
    draw.ellipse(bounds, outline='#b8ff65', width=round(4 * scale))
    return highlighted


render_scan = st.cache_data(max_entries=8,show_spinner=False)(scan_gif)


@st.cache_resource(show_spinner='Đang nạp mô hình…')
def load_detector(path, modified):
    import torch
    from pcb.engine import Detector
    torch.set_num_threads(4)
    return Detector.load(path,'cpu')

@st.cache_data
def get_samples(stamp):
    return manifest()

checkpoints=sorted(p for p in (ROOT/'outputs').glob('*/weights/best.pt') if p.parents[1].name!='integration_pair')
single_checkpoints=[p for p in checkpoints if run_config(p.parents[1]).get('mode')=='test']
single_checkpoints.sort(key=lambda p:(p.parents[1].name!='yolov8n_gpu_full_01',
                                     bool(run_config(p.parents[1]).get('smoke_test')),str(p)))
with st.sidebar:
    st.markdown(
        '<div class="pcb-sidebar-brand"><span class="pcb-brand-mark" aria-hidden="true"></span>'
        '<div class="pcb-brand-name">PCB <span>LAB</span></div></div>', unsafe_allow_html=True)
    st.subheader('Thiết lập phát hiện')
    checkpoint=st.selectbox('Mô hình',single_checkpoints,
        format_func=lambda p:f'{run_label(p.parents[1])} · {p.parents[1].name}') if single_checkpoints else None
    if checkpoint:
        run=checkpoint.parents[1]
        cfg=run_config(run)
        status_file=run/'status.json'
        epochs=read_json(status_file).get('epochs') if status_file.exists() else None
        if epochs is not None:
            st.caption(f'Đã huấn luyện {epochs} epoch')
        st.caption(f'Đầu vào {cfg.get("imgsz","—")} × {cfg.get("imgsz","—")} · Giao diện chạy CPU')
        with st.expander('Thông tin mô hình'):
            st.caption(run.name)
            st.write(f'**Kiến trúc:** {cfg.get("model","—").upper()}')
            st.write(f'**Biểu diễn ảnh:** {MODES.get(cfg.get("mode"),"—")}')
            st.write(f'**Batch huấn luyện:** {cfg.get("batch","—")} · **Seed:** {cfg.get("seed","—")}')
            st.caption('best.pt được chọn theo mAP@50–95 Validation.')
    confidence=st.slider('Ngưỡng điểm tin cậy',0.05,0.95,0.25,0.05)
    st.caption('Tăng ngưỡng để giảm các dự đoán có điểm thấp. Ngưỡng cần được lựa chọn bằng Validation khi báo cáo.')
    st.divider()
    st.markdown('**Chú giải loại lỗi**')
    legend=''.join(f'<li style="--defect-color:{color}"><i aria-hidden="true"></i>'
        f'<span>{vi} <small>· {en}</small></span></li>' for color,vi,en in zip(COLORS,VI,NAMES))
    st.markdown(f'<ul class="pcb-legend">{legend}</ul>',unsafe_allow_html=True)

detect_tab,results_tab,data_tab=st.tabs(['Phát hiện','Kết quả thực nghiệm','Dữ liệu và hướng dẫn'])
with detect_tab:
    with st.container(key='pcb_source'):
        st.subheader('Chọn ảnh PCB')
        source=st.radio('Nguồn ảnh',['Tải ảnh của bạn','Mẫu DeepPCB'],index=1,horizontal=True)
        tested=None
        reference=None
        enhance_contrast=False
        sample_id='uploaded'
        if source=='Mẫu DeepPCB':
            manifest_file=ROOT/'data/processed/manifest.json'
            if manifest_file.exists():
                rows=get_samples(manifest_file.stat().st_mtime)
                c1,c2=st.columns([1,3])
                split=c1.selectbox('Tập dữ liệu',['val','test','train'],format_func=SPLITS.get)
                sample=c2.selectbox('Chọn ảnh',[r for r in rows if r['split']==split],
                    format_func=lambda r:f'{r["id"]} · {len(r["boxes"])} lỗi trong nhãn gốc')
                if sample is not None:
                    sample_id=sample['id']
                    with Image.open(project_path(sample['template'])) as im:
                        reference=im.convert('RGB')
                    with Image.open(project_path(sample['tested'])) as im:
                        tested=im.convert('RGB')
            else:
                st.info('Chưa có danh sách dữ liệu. Chạy CHUAN_BI_DU_LIEU.cmd để dùng ảnh mẫu DeepPCB.')
        else:
            uploaded=st.file_uploader('Ảnh PCB đầu vào',type=['png','jpg','jpeg','bmp','tif'])
            if uploaded is not None:
                enhance_contrast=st.checkbox('Tăng tương phản ảnh',
                    help='Thử khi đường đồng và nền khó phân biệt. Khung dự đoán vẫn được vẽ trên ảnh gốc.')
                try:
                    with Image.open(uploaded) as im:
                        tested=im.convert('RGB')
                except (OSError,ValueError,ImportError):
                    st.error('Không đọc được ảnh. Hãy chọn một tệp ảnh hợp lệ.')

    if tested is not None:
        if reference is not None:
            reference_column,left,right=st.columns(3,gap='medium')
            reference_panel=reference_column.container(key='pcb_reference')
            reference_panel.markdown('**Ảnh mẫu chuẩn**')
            reference_panel.image(reference,caption=f'Cặp {sample_id} · Ảnh tham chiếu',width='stretch')
            st.caption('Ảnh mẫu chuẩn dùng để đối chiếu; mô hình vẫn dự đoán từ ảnh kiểm tra.')
        else:
            left,right=st.columns(2,gap='medium')
        input_panel=left.container(key='pcb_input')
        input_panel.markdown('**Ảnh kiểm tra**' if reference is not None else '**Ảnh tải lên**')
        input_panel.image(tested,caption=f'{tested.width:,} × {tested.height:,} px',width='stretch')
        result_panel=right.container(key='pcb_result')
        result_panel.markdown('**Kết quả phát hiện**')
        result=None
        if checkpoint is None:
            st.info('Chưa có mô hình dùng ảnh kiểm tra. Chạy HUAN_LUYEN_YOLO.cmd để tạo checkpoint.')
        else:
            checkpoint_name=str(checkpoint.relative_to(ROOT))
            modified=checkpoint.stat().st_mtime_ns
            prediction_key=(sample_id,checkpoint_name,modified,confidence,enhance_contrast,
                tested.size,hash(tested.tobytes()))
            if cfg.get('smoke_test'):
                st.warning('Đây là mô hình chạy thử trên tập nhỏ.')
            saved=st.session_state.get('prediction')
            if saved and saved.get('key')==prediction_key and 'tiles_scanned' in saved:
                result=saved
            else:
                st.session_state.pop('prediction',None)
                with result_panel, st.spinner('Đang phát hiện lỗi…'):
                    try:
                        detector=load_detector(str(checkpoint),modified)
                        gray_image=tested.convert('L')
                        if enhance_contrast:
                            gray_image=ImageOps.autocontrast(gray_image,cutoff=1)
                        predictions,ms,tile_count=detector.predict_image(np.asarray(gray_image),confidence)
                        result=dict(key=prediction_key,image=annotate(tested,predictions),
                            predictions=predictions,processing_ms=ms,sample_id=sample_id,
                            checkpoint=checkpoint_name,confidence=confidence,tiles_scanned=tile_count,
                            contrast_enhanced=enhance_contrast)
                        st.session_state['prediction']=result
                    except Exception as error:
                        st.error(f'Phát hiện thất bại: {error}')

        if result is not None:
            preview=result_panel.empty()
            preview_image=result['image']
            preds=result['predictions']
            preview_caption=f'{len(preds)} vùng dự đoán · Ngưỡng {confidence:.2f}'
            tile_count=result.get('tiles_scanned',1)
            if tile_count>1:
                st.caption(f'Đã quét {tile_count} vùng chồng lấn trên ảnh toàn bo.')
            if not preds:
                st.info('Không phát hiện lỗi ở ngưỡng đã chọn.')
            with st.container(key='pcb_review'):
                st.subheader('Chi tiết dự đoán và tải kết quả')
                counts=pd.Series([p['class_vi'] for p in preds],dtype=str).value_counts()
                cols=st.columns(4)
                for col,label,value in zip(cols,['Vùng lỗi được dự đoán','Loại lỗi xuất hiện','Tin cậy trung bình','Thời gian xử lý CPU'],
                    [len(preds),len(counts),f'{np.mean([p["score"] for p in preds])*100:.1f}%' if preds else '—',f'{result["processing_ms"]:.0f} ms']):
                    col.metric(label,value)
                records=[dict(STT=n,Loại_lỗi=p['class_vi'],Tin_cậy=round(p['score']*100,2),
                    **{key:round(v,1) for key,v in zip(['x1','y1','x2','y2'],p['bbox_xyxy'])}) for n,p in enumerate(preds,1)]
                if preds:
                    st.caption('Chọn một dòng trong bảng để khoanh tròn vị trí lỗi trên ảnh kết quả.')
                    table=st.dataframe(
                        pd.DataFrame(records),hide_index=True,width='stretch',
                        key=f'predictions_{hash(result["key"])}',
                        on_select='rerun',selection_mode='single-row',
                        column_config={'Loại_lỗi':'Loại lỗi','Tin_cậy':st.column_config.ProgressColumn(
                            'Tin cậy (%)',min_value=0,max_value=100,format='%.2f%%')})
                    if table.selection.rows:
                        row=table.selection.rows[0]
                        if 0 <= row < len(preds):
                            selected=preds[row]
                            preview_image=highlight_detection(preview_image,selected['bbox_xyxy'])
                            preview_caption=f'Lỗi {row + 1} · {selected["class_vi"]} · {selected["score"]:.1%}'
                png=io.BytesIO()
                preview_image.save(png,format='PNG')
                show_scan=st.checkbox('Hiệu ứng quét trên ảnh kết quả',value=True)
                animation=None
                if show_scan:
                    original_png=io.BytesIO()
                    tested.save(original_png,format='PNG')
                    with st.spinner('Đang tạo hiệu ứng quét…'):
                        animation=render_scan(original_png.getvalue(),png.getvalue())
                c1,c2=st.columns(2)
                c1.download_button('Tải ảnh kết quả',animation if animation else png.getvalue(),
                    file_name=f'{sample_id}_result.gif' if animation else f'{sample_id}_result.png',
                    mime='image/gif' if animation else 'image/png',type='primary')
                c1.caption('GIF · quét một lượt rồi dừng ở kết quả' if animation else 'PNG · ảnh kết quả')
                payload={k:v for k,v in result.items() if k not in ['image','key']}
                c2.download_button('Tải kết quả JSON',json.dumps(payload,ensure_ascii=False,indent=2),file_name=f'{sample_id}_prediction.json',mime='application/json')
                if preds:
                    with st.expander('Phân bố lỗi và thời gian xử lý'):
                        if len(counts):
                            chart=counts.rename_axis('Loại lỗi').reset_index(name='Số lượng')
                            st.bar_chart(chart,x='Loại lỗi',y='Số lượng',horizontal=True,color='#0284a8',height=280)
                        st.caption('Thời gian xử lý không tính nạp mô hình và đọc tệp.')
            preview.image(animation if animation else preview_image,caption=preview_caption,width='stretch')

    else:
        st.markdown(
            '<div class="pcb-empty"><div class="pcb-empty-target" aria-hidden="true"></div>'
            '<strong>Sẵn sàng phân tích ảnh PCB</strong>'
            '<p>Tải một ảnh ở phía trên hoặc chọn Mẫu DeepPCB để thử phát hiện. '
            'Ảnh đầu vào và kết quả khoanh vùng lỗi sẽ xuất hiện tại đây.</p></div>',
            unsafe_allow_html=True)

with results_tab:
    with st.container(key='pcb_compare'):
        st.subheader('So sánh kết quả thực nghiệm')
        st.caption('Chọn mô hình bằng Validation; dùng Test để báo cáo khả năng tổng quát hóa. Các số liệu được đọc từ kết quả đã lưu.')
        compare_split=st.radio('Tập đánh giá',['val','test','train'],format_func=SPLITS.get,horizontal=True)
        eligible=sorted(path.parent for path in (ROOT/'outputs').glob('*/config.json')
            if not run_config(path.parent).get('smoke_test'))
        selected_runs=st.multiselect('Lượt chạy so sánh',eligible,
            default=[run for run in eligible if '_full_' in run.name],
            format_func=lambda run:f'{run_label(run)} · {run.name}')
        comparison=[]
        for run in selected_runs:
            path=run/('validation.json' if compare_split=='val' else f'evaluation_{compare_split}/metrics.json')
            if not path.exists(): continue
            m=read_json(path)
            if m.get('limited_evaluation') or m.get('checkpoint_smoke_test'): continue
            comparison.append({'Mô hình':run_label(run),'Lượt chạy':run.name,
                **{key:m[key]*100 for key in ['precision','recall','f1','map50','map50_95']},'FP':m['fp'],'FN':m['fn'],'Cặp ảnh':m['images']})
        formats={key:st.column_config.NumberColumn(label,format='%.2f%%') for key,label in
            {'precision':'Precision','recall':'Recall','f1':'F1-score','map50':'mAP@50','map50_95':'mAP@50–95'}.items()}
        if comparison:
            frame=pd.DataFrame(comparison)
            st.dataframe(frame,hide_index=True,width='stretch',column_config=formats)
            chart=frame.rename(columns={'f1':'F1-score','map50_95':'mAP@50–95'})
            if chart['Mô hình'].duplicated().any():chart['Mô hình']=chart['Mô hình']+' · '+chart['Lượt chạy']
            st.bar_chart(chart,x='Mô hình',y=['F1-score','mAP@50–95'],color=['#0c9e85','#4f8df7'],stack=False,height=310,y_label='Điểm (%)')
        else: st.info('Chưa có báo cáo đầy đủ cho tập được chọn.')
    st.divider()
    st.subheader('Chi tiết một lượt đánh giá')
    reports=sorted((ROOT/'outputs').glob('*/evaluation_*/metrics.json'))
    reports.sort(key=lambda p:(p.parent.parent.name!='yolov8n_pair_full_01',p.parent.name!='evaluation_test',str(p)))
    if not reports:
        st.info('Chưa có kết quả đánh giá. Xem lệnh evaluate trong README.md.')
    else:
        report=st.selectbox('Báo cáo đánh giá',reports,format_func=lambda p:str(p.relative_to(ROOT/'outputs')))
        stats=read_json(report)
        if stats.get('checkpoint_smoke_test') or stats.get('limited_evaluation'):
            st.warning('Báo cáo kiểm tra trên tập nhỏ hoặc checkpoint smoke test; không phải kết quả thực nghiệm đầy đủ.')
        cols=st.columns(5)
        for col,key,label in zip(cols,['precision','recall','f1','map50','map50_95'],['Precision','Recall','F1-score','mAP@50','mAP@50–95']):
            col.metric(label,f'{stats[key]*100:.2f}%')
        st.caption(f'{stats["images"]} cặp ảnh · {stats["ground_truth"]} nhãn · TP {stats["tp"]} / FP {stats["fp"]} / FN {stats["fn"]} · IoU {stats["match_iou"]} · Confidence {stats["confidence"]}')
        detail=pd.DataFrame([{'Loại lỗi':VI[NAMES.index(c['name'])],'Nhãn gốc':c['ground_truth'],'TP':c['tp'],'FP':c['fp'],'FN':c['fn'],
            **{key:c[key]*100 if c[key] is not None else None for key in ['precision','recall','f1','ap50_95']}} for c in stats['per_class']])
        st.dataframe(detail,hide_index=True,width='stretch',column_config={**formats,'ap50_95':st.column_config.NumberColumn('AP@50–95',format='%.2f%%')})
        if stats.get('mean_preprocess_inference_postprocess_ms') is not None:
            st.caption(f'Thời gian vòng đánh giá: {stats["mean_preprocess_inference_postprocess_ms"]:.2f} ms / ảnh kiểm tra; không gồm đọc tệp, hiển thị và xuất kết quả. Khác thời gian giao diện chạy CPU.')
    curves=sorted((ROOT/'outputs').glob('*/learning_curves.png'))
    curves.sort(key=lambda p:(p.parent.name!='yolov8n_pair_full_01',str(p)))
    if curves:
        curve=st.selectbox('Quá trình học',curves,format_func=lambda p:p.parent.name)
        st.image(str(curve),width='stretch')
        history_file=curve.parent/'history.csv'
        if history_file.exists():
            history=pd.read_csv(history_file)
            best=history.loc[history['map50_95'].idxmax()]
            st.caption(f'Đã chạy {len(history)} epoch · Tốt nhất tại epoch {int(best["epoch"])} · Validation mAP@50–95 {best["map50_95"]*100:.2f}%')

with data_tab:
    st.subheader('Bộ dữ liệu DeepPCB')
    st.write('Ảnh PCB theo cặp mẫu chuẩn và kiểm tra, kích thước 640 × 640. Mỗi lỗi được gán hộp giới hạn và một trong sáu loại lỗi.')
    audit=ROOT/'data/processed/audit.json'
    if audit.exists():
        stats=read_json(audit)
        pairs=sum(x['pairs'] for x in stats['splits'].values())
        boxes=sum(x['boxes'] for x in stats['splits'].values())
        for col,label,value in zip(st.columns(4),['Cặp ảnh','Tổng ảnh','Hộp nhãn','Loại lỗi'],[f'{pairs:,}',f'{pairs*2:,}',f'{boxes:,}',len(NAMES)]):
            col.metric(label,value)
        st.dataframe(pd.DataFrame([{'Tập':SPLITS[s],'Cặp ảnh':x['pairs'],'Số ảnh':x['pairs']*2,'Hộp lỗi':x['boxes']} for s,x in stats['splits'].items()]),hide_index=True,width='stretch')
        distribution=pd.DataFrame([{'Loại lỗi':vi,**{SPLITS[s]:x['per_class'][en] for s,x in stats['splits'].items()}} for en,vi in zip(NAMES,VI)])
        st.markdown('**Phân bố hộp nhãn theo loại lỗi**')
        st.bar_chart(distribution,x='Loại lỗi',y=['Train','Validation','Test'],color=['#02a4be','#0c9e85','#8b5cf6'],stack=False,height=300)
        st.caption('Test giữ cách chia gốc. Validation tách theo cặp từ Train gốc; không phải kiểm thử độc lập theo thiết kế mạch.')
    with st.expander('Cách đọc kết quả'):
        st.markdown('''- **Precision:** tỷ lệ dự đoán ghép đúng trong các dự đoán được giữ lại.
- **Recall:** tỷ lệ lỗi gốc được phát hiện.
- **F1-score:** cân bằng Precision và Recall tại ngưỡng đã chọn.
- **mAP@50–95:** đánh giá phát hiện và định vị qua nhiều mức IoU.
- **Ảnh hiệu:** biểu diễn vùng sai khác, chưa phải kết quả phân loại lỗi.''')
    with st.expander('Hướng dẫn chạy project'):
        st.markdown('''**Trình tự thực hiện**
1. Chạy `CHUAN_BI_DU_LIEU.cmd` để kiểm tra và chia DeepPCB.
2. Chạy `HUAN_LUYEN_YOLO.cmd` để huấn luyện mô hình cơ sở.
3. Theo README để chạy ảnh hiệu, hai kênh và so sánh YOLOv8n / YOLO11n.
4. Chọn checkpoint bằng Validation; đánh giá Train và Test sau khi chốt cấu hình.
5. Lưu bảng chỉ số, đồ thị và ảnh kết quả để hoàn thành Chương 3.
''')
    st.markdown('[DeepPCB chính thức](https://github.com/tangsanli5201/DeepPCB) · [Ultralytics](https://docs.ultralytics.com/)')

