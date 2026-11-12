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

st.set_page_config(page_title='PCB Lab | Phát hiện lỗi mạch in',page_icon='🔬',layout='wide')
st.markdown('''<style>
:root {
    --pcb-ink:#112b43;
    --pcb-muted:#48647b;
    --pcb-accent:#087d78;
    --pcb-line:#d6e3ee;
    --pcb-shadow:0 8px 26px rgba(24,65,97,.07);
}
.stApp {background:linear-gradient(160deg,#edf5fc 0%,#f7fafc 42%,#eff6f8 100%);color:var(--pcb-ink)}
[data-testid="stHeader"] {background:rgba(247,250,252,.92)}
.block-container {max-width:1440px;padding-top:3.25rem;padding-bottom:3.5rem}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#fff 0%,#f2f8fb 100%);border-right:1px solid var(--pcb-line)}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {min-width:0}
h1,h2,h3 {color:var(--pcb-ink);letter-spacing:-.032em}
[data-testid="stCaptionContainer"] p {color:var(--pcb-muted)}
.st-key-pcb_hero {
    position:relative; overflow:hidden; border:1px solid #cbdfee; border-radius:26px;
    background:radial-gradient(ellipse at 91% 4%,#c3edeb 0%,transparent 47%),
      linear-gradient(120deg,#fff 0%,#eef7ff 54%,#dbeef8 100%);
    box-shadow:0 14px 42px rgba(36,83,119,.11); padding:clamp(1.3rem,3vw,2.8rem);
}
.st-key-pcb_hero h1 {font-size:clamp(2.8rem,5vw,4.8rem);line-height:1;padding-top:.14em;padding-bottom:.14em}
.pcb-eyebrow {font-size:.72rem;letter-spacing:.15em;text-transform:uppercase;font-weight:750;color:#096767}
.pcb-lead {color:#355873;font-size:clamp(.98rem,1.5vw,1.2rem);max-width:31rem;line-height:1.55;margin:0}
.pcb-hero-figure {
    margin:0; background:rgba(255,255,255,.8); padding:.6rem; border:1px solid #c4dbe8;
    border-radius:18px; box-shadow:0 12px 24px rgba(30,77,105,.13);
}
.pcb-hero-figure img {display:block;width:100%;height:clamp(126px,15vw,196px);object-fit:cover;
    object-position:50% 45%;border-radius:11px;background:#fff}
.pcb-hero-figure figcaption {color:#33576f;padding:.5rem .4rem .1rem;font-size:.75rem;font-weight:650}
[data-testid="stTabs"] [role="tablist"] {
    gap:.35rem; margin:.35rem 0 .85rem; padding:.32rem; width:fit-content;max-width:100%;border-bottom:0;
    border:1px solid var(--pcb-line);background:#fff;border-radius:14px;box-shadow:var(--pcb-shadow);
}
[data-testid="stTabs"] [role="tab"] {font-weight:650;padding:.55rem .9rem;border-radius:9px;color:#395b74}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {background:#e0f4f3;color:#07615c}
[data-testid="stTabs"] .react-aria-SelectionIndicator,[data-testid="stTabs"] [data-baseweb="tab-highlight"] {display:none}
.st-key-pcb_source,.st-key-pcb_input,.st-key-pcb_result,.st-key-pcb_review,.st-key-pcb_compare {
    background:#fff;border:1px solid var(--pcb-line);border-radius:18px;
    box-shadow:var(--pcb-shadow);padding:1.05rem 1.25rem;min-width:0;
}
.st-key-pcb_input,.st-key-pcb_result {height:100%}
.st-key-pcb_result {border-top:3px solid var(--pcb-accent)}
.st-key-pcb_source [data-testid="stFileUploaderDropzone"] {
    background:#f0f7fc;border:1px dashed #9dbcd2;border-radius:12px;
}
[data-testid="stImage"] img {border-radius:11px}
.st-key-pcb_input [data-testid="stImage"] img,.st-key-pcb_result [data-testid="stImage"] img {
    border:1px solid var(--pcb-line); display:block;
}
div[data-testid="stMetric"] {
    background:linear-gradient(150deg,#fff,#f0f8fc);border:1px solid #dce8f0;border-radius:13px;
    padding:.85rem .95rem;height:100%;
}
[data-testid="stMetricLabel"] p {color:#38566d;font-weight:630}
[data-testid="stMetricValue"] {color:#076965;font-variant-numeric:tabular-nums}
[data-testid="stDataFrame"],[data-testid="stExpander"] details {border-radius:12px}
.pcb-legend {display:grid;gap:.4rem;margin:.35rem 0 0;padding:0;list-style:none}
.pcb-legend li {display:grid;grid-template-columns:10px 1fr;gap:.5rem;align-items:baseline;
    font-size:.84rem;padding:.36rem .5rem;background:#fff;border:1px solid #e0eaf2;border-radius:8px}
.pcb-legend i {width:9px;height:9px;border-radius:50%;background:var(--defect-color);border:1px solid #53697f66}
.pcb-legend small {color:#526b81;font-size:.72rem}
@media(max-width:720px) {
    .block-container {padding-top:3rem}
    .st-key-pcb_hero {border-radius:18px}
    .st-key-pcb_source,.st-key-pcb_input,.st-key-pcb_result,.st-key-pcb_review,.st-key-pcb_compare {padding:.8rem}
    [data-testid="stTabs"] [role="tab"] {padding:.45rem .55rem}
}
</style>''',unsafe_allow_html=True)

hero_preview=ROOT/'outputs/yolov8n_gpu_full_01/evaluation_test/previews/20085291.png'
with st.container(key='pcb_hero'):
    introduction,illustration=st.columns([1.5,1],vertical_alignment='center')
    with introduction:
        st.markdown('<div class="pcb-eyebrow">Kiểm tra ngoại quan mạch in</div>',unsafe_allow_html=True)
        st.title('PCB Lab')
        st.markdown('<p class="pcb-lead">Tải một ảnh PCB để phát hiện lỗi và xem vị trí dự đoán trên bo mạch.</p>',
            unsafe_allow_html=True)
    if hero_preview.is_file():
        encoded=base64.b64encode(hero_preview.read_bytes()).decode('ascii')
        illustration.markdown(f'<figure class="pcb-hero-figure"><img src="data:image/png;base64,{encoded}" '
            'alt="Ảnh mạch in DeepPCB có các khung lỗi dự đoán"><figcaption>Mẫu DeepPCB · Ảnh đã khoanh vùng lỗi</figcaption></figure>',
            unsafe_allow_html=True)

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
    draw.ellipse(bounds, outline='#172b43', width=round(7 * scale))
    draw.ellipse(bounds, outline='#ffdf42', width=round(4 * scale))
    return highlighted


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
        left,right=st.columns(2,gap='medium')
        input_panel=left.container(key='pcb_input')
        input_panel.markdown('**Ảnh đầu vào**')
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
                c1,c2=st.columns(2)
                c1.download_button('Tải ảnh kết quả',png.getvalue(),file_name=f'{sample_id}_result.png',mime='image/png')
                payload={k:v for k,v in result.items() if k not in ['image','key']}
                c2.download_button('Tải kết quả JSON',json.dumps(payload,ensure_ascii=False,indent=2),file_name=f'{sample_id}_prediction.json',mime='application/json')
                if preds:
                    with st.expander('Phân bố lỗi và thời gian xử lý'):
                        if len(counts):
                            chart=counts.rename_axis('Loại lỗi').reset_index(name='Số lượng')
                            st.bar_chart(chart,x='Loại lỗi',y='Số lượng',horizontal=True,color='#0f766e',height=280)
                        st.caption('Thời gian xử lý không tính nạp mô hình và đọc tệp.')
            preview.image(preview_image,caption=preview_caption,width='stretch')

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
            st.bar_chart(chart,x='Mô hình',y=['F1-score','mAP@50–95'],color=['#0f766e','#3b82f6'],stack=False,height=310,y_label='Điểm (%)')
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
        st.bar_chart(distribution,x='Loại lỗi',y=['Train','Validation','Test'],color=['#0f766e','#3b82f6','#f59e0b'],stack=False,height=300)
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

