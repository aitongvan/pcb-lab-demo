"""Verify prediction, exports and invalid inputs in the deployment environment."""
from pathlib import Path
import sys
import json
import io
from PIL import Image
from fastapi.testclient import TestClient
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import app

client = TestClient(app)
content = (ROOT / 'public/samples/20085291.jpg').read_bytes()
response = client.post('/api/predict', files={'file': ('pcb.jpg', content, 'image/jpeg')})
assert response.status_code == 200, response.text
result = response.json()
assert result['predictions']
for animation in (True, False):
    exported = client.post('/api/export', files={'file': ('pcb.jpg', content, 'image/jpeg')},
                           data={'predictions': json.dumps(result['predictions']),
                                 'animation': str(animation).lower(), 'selected': '0'})
    assert exported.status_code == 200, exported.text
    with Image.open(io.BytesIO(exported.content)) as output:
        assert output.size == (640, 640)
        if animation:
            assert output.n_frames == 26 and 'loop' not in output.info
        else:
            assert output.format == 'PNG'
assert client.post('/api/predict', files={'file': ('bad.jpg', b'invalid')}).status_code == 422
assert client.post('/api/predict', files={'file': ('pcb.jpg', content)}, data={'confidence': 2}).status_code == 422
assert client.get('/').status_code == 200
assert client.get('/static/samples.json').status_code == 200
print('All numerical comparisons, prediction API, exports and input validation passed.')
