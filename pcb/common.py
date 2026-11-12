import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole']
VI = ['Đứt mạch', 'Chập mạch', 'Khuyết cạnh', 'Gai thừa', 'Đồng thừa', 'Lỗ kim']
COLORS = ['#f87171', '#fbbf24', '#60a5fa', '#a78bfa', '#34d399', '#fb923c']

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path

