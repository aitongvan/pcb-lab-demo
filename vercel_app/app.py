"""PCB Lab HTTP API and frontend entrypoint for Vercel."""
import io
import json
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, ValidationError

from inference import MAX_IMAGE_BYTES, NAMES, VI, annotate, predict, read_image, available_models
from rendering import scan_gif

ROOT = Path(__file__).resolve().parent
app = FastAPI(title='PCB Lab', docs_url=None, redoc_url=None)
# Vercel serves public assets from its CDN and omits them from the function.
app.mount('/static', StaticFiles(directory=ROOT / 'public', check_dir=False), name='static')


@app.get('/')
def home():
    index = ROOT / 'public/index.html'
    return FileResponse(index) if index.exists() else RedirectResponse('/index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok', 'model': 'YOLOv8n', 'runtime': 'ONNX Runtime CPU'}


@app.get('/api/models')
def models():
    return {'models': available_models()}


@app.post('/api/compare')
async def compare(file: UploadFile = File(...), confidence: float = Form(.05),
                  enhance: bool = Form(False)):
    image = await uploaded_image(file)
    if not all(item['available'] for item in available_models()):
        raise HTTPException(503, 'Chưa có trọng số YOLO11n một ảnh. Chức năng so sánh sẽ mở khi mô hình được bổ sung.')
    try:
        first = await run_in_threadpool(predict, image, confidence, enhance, 'yolov8n')
        second = await run_in_threadpool(predict, image, confidence, enhance, 'yolo11n')
        return {'results': [first, second], 'timing_note': 'Hai mô hình chạy lần lượt trên CPU, không tính thời gian nạp và làm nóng mô hình. Số vùng và điểm tin cậy không xác định độ chính xác khi ảnh chưa có nhãn.'}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logging.exception('PCB comparison failed')
        raise HTTPException(500, 'Chưa so sánh được mô hình. Hãy thử lại sau.') from error


async def uploaded_image(file):
    data = await file.read(MAX_IMAGE_BYTES + 1)
    try:
        return read_image(data)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post('/api/predict')
async def detect(file: UploadFile = File(...), confidence: float = Form(.05),
                 enhance: bool = Form(False)):
    image = await uploaded_image(file)
    try:
        return await run_in_threadpool(predict, image, confidence, enhance)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    except Exception as error:
        logging.exception('PCB prediction failed')
        raise HTTPException(500, 'Chưa xử lý được ảnh. Hãy thử lại sau ít giây.') from error


class Detection(BaseModel):
    class_id: int = Field(ge=0, lt=6)
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    bbox_xyxy: list[float] = Field(min_length=4, max_length=4)


@app.post('/api/export')
async def export(file: UploadFile = File(...), predictions: str = Form(...),
                 animation: bool = Form(True), selected: int = Form(-1)):
    image = await uploaded_image(file)
    try:
        values = json.loads(predictions)
        if not isinstance(values, list) or len(values) > 5000:
            raise ValueError('Danh sách dự đoán không hợp lệ.')
        parsed = [Detection.model_validate(value) for value in values]
        records = []
        for value in parsed:
            x1, y1, x2, y2 = value.bbox_xyxy
            if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                raise ValueError('Tọa độ khung lỗi không hợp lệ.')
            records.append(dict(value.model_dump(), class_name=NAMES[value.class_id],
                                class_vi=VI[value.class_id]))
    except (ValueError, TypeError, ValidationError) as error:
        raise HTTPException(422, 'Dữ liệu xuất ảnh không hợp lệ.') from error

    def render():
        result = annotate(image, records, selected)
        output = io.BytesIO()
        result.save(output, format='PNG')
        if animation:
            original = io.BytesIO()
            image.save(original, format='PNG')
            return scan_gif(original.getvalue(), output.getvalue())
        return output.getvalue()

    content = await run_in_threadpool(render)
    if len(content) > 4 * 1024 * 1024:
        raise HTTPException(413, 'Ảnh xuất quá lớn. Hãy giảm kích thước ảnh đầu vào.')
    extension = 'gif' if animation else 'png'
    return Response(content, media_type=f'image/{extension}', headers={
        'Content-Disposition': f'attachment; filename="pcb_result.{extension}"',
        'Cache-Control': 'no-store',
    })
