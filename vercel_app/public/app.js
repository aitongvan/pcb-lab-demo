'use strict';

const $ = (id) => document.getElementById(id);
const names = ['open', 'short', 'mousebite', 'spur', 'copper', 'pin-hole'];
const vietnamese = ['Đứt mạch', 'Chập mạch', 'Khuyết cạnh', 'Gai thừa', 'Đồng thừa', 'Lỗ kim'];
const colors = ['#f87171', '#fbbf24', '#60a5fa', '#a78bfa', '#34d399', '#fb923c'];
const modes = { test: 'Ảnh kiểm tra', diff: 'Ảnh hiệu', pair: 'Hai kênh' };
const splits = { train: 'Train', val: 'Validation', test: 'Test' };
const state = { file: null, image: null, objectUrl: null, result: null, selected: -1,
  samples: [], experiments: [], controller: null, imageVersion: 0, requestVersion: 0 };
let thresholdTimer;
let animationTimer;

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function notice(message = '', error = false) {
  $('notice').textContent = message;
  $('notice').classList.toggle('error', error);
  $('notice').hidden = !message;
}

function metric(label, value) {
  const card = element('div', undefined, 'metric');
  card.append(element('span', label), element('strong', value));
  return card;
}

function formatPercent(value) {
  return value == null ? '—' : `${(value * 100).toFixed(2)}%`;
}

function addCells(row, values) {
  values.forEach((value) => row.append(element('td', String(value))));
}

function showPanel(button) {
  document.querySelectorAll('.tab').forEach((tab) => {
    const active = tab === button;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    $(tab.dataset.panel).hidden = !active;
  });
}

document.querySelectorAll('.tab').forEach((button) => {
  button.addEventListener('click', () => showPanel(button));
  button.addEventListener('keydown', (event) => {
    const tabs = [...document.querySelectorAll('.tab')];
    if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    let next = tabs.indexOf(button);
    if (event.key === 'ArrowLeft') next = (next + tabs.length - 1) % tabs.length;
    if (event.key === 'ArrowRight') next = (next + 1) % tabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = tabs.length - 1;
    showPanel(tabs[next]);
    tabs[next].focus();
  });
});

names.forEach((name, index) => {
  const item = element('li');
  const dot = element('i');
  dot.style.setProperty('--color', colors[index]);
  dot.setAttribute('aria-hidden', 'true');
  const label = element('span', `${vietnamese[index]} `);
  label.append(element('small', `· ${name}`));
  item.append(dot, label);
  $('legend').append(item);
});

async function responseJson(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = typeof data.detail === 'string' ? data.detail : 'Chưa xử lý được yêu cầu. Hãy thử lại.';
    throw new Error(message);
  }
  return data;
}

function cancelPrediction() {
  state.controller?.abort();
  state.requestVersion += 1;
  clearTimeout(thresholdTimer);
  clearTimeout(animationTimer);
  $('scanStage').classList.remove('scanning');
  $('processingOverlay').hidden = true;
}

async function loadFile(file) {
  const version = ++state.imageVersion;
  cancelPrediction();
  state.file = null;
  state.result = null;
  $('regionControls').hidden = true;
  $('review').hidden = true;
  $('imageGrid').hidden = true;
  $('imageDimensions').textContent = '';
  if (!file) return;
  if (file.size > 4 * 1024 * 1024) {
    notice('Chọn ảnh có dung lượng tối đa 4 MB.', true);
    return;
  }
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.src = url;
  try {
    await image.decode();
    if (version !== state.imageVersion) {
      URL.revokeObjectURL(url);
      return;
    }
    if (image.naturalWidth * image.naturalHeight > 24_000_000) {
      throw new Error('Chọn ảnh có tối đa 24 triệu điểm ảnh.');
    }
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = url;
    state.file = file;
    state.image = image;
    $('inputImage').src = url;
    $('resultBase').src = url;
    const canvas = $('resultCanvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $('imageDimensions').textContent = `${image.naturalWidth.toLocaleString()} × ${image.naturalHeight.toLocaleString()} px`;
    $('imageGrid').hidden = false;
    await analyze();
  } catch (error) {
    URL.revokeObjectURL(url);
    if (version === state.imageVersion) notice(error.message || 'Không đọc được ảnh.', true);
  }
}

async function analyze() {
  if (!state.file) return;
  cancelPrediction();
  const version = state.requestVersion;
  state.controller = new AbortController();
  state.result = null;
  $('regionControls').hidden = true;
  $('review').hidden = true;
  $('predictionCount').textContent = '';
  $('selectedLabel').textContent = '';
  $('resultCanvas').getContext('2d').clearRect(0, 0, $('resultCanvas').width, $('resultCanvas').height);
  $('processingOverlay').hidden = false;
  notice();
  const body = new FormData();
  body.append('file', state.file);
  body.append('confidence', $('confidence').value);
  body.append('enhance', String($('enhance').checked));
  try {
    const response = await fetch('/api/predict', { method: 'POST', body, signal: state.controller.signal });
    const result = await responseJson(response);
    if (version !== state.requestVersion) return;
    state.result = result;
    state.selected = -1;
    $('processingOverlay').hidden = true;
    renderPredictions();
    drawResult(true);
  } catch (error) {
    if (error.name === 'AbortError' || version !== state.requestVersion) return;
    $('processingOverlay').hidden = true;
    notice(error.message || 'Mất kết nối khi phân tích ảnh. Hãy thử lại.', true);
  }
}

function drawResult(animate = false) {
  if (!state.result) return;
  const canvas = $('resultCanvas');
  const context = canvas.getContext('2d');
  context.drawImage(state.image, 0, 0);
  const scale = Math.max(1, Math.max(canvas.width, canvas.height) / 900);
  context.font = `${Math.round(12 * scale)}px sans-serif`;
  state.result.predictions.forEach((prediction, index) => {
    const [x1, y1, x2, y2] = prediction.bbox_xyxy;
    const color = colors[prediction.class_id];
    context.strokeStyle = color;
    context.lineWidth = 2 * scale;
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${prediction.class_name} ${prediction.score.toFixed(2)}`;
    const textWidth = context.measureText(text).width + 6 * scale;
    const textHeight = 17 * scale;
    const top = Math.max(0, y1 - textHeight);
    context.fillStyle = '#08131c';
    context.fillRect(x1, top, textWidth, textHeight);
    context.fillStyle = color;
    context.fillText(text, x1 + 3 * scale, top + 12 * scale);
    if (index === state.selected) {
      const radius = Math.max(15 * scale, Math.hypot(x2 - x1, y2 - y1) / 2 + 5 * scale);
      context.beginPath();
      context.arc((x1 + x2) / 2, (y1 + y2) / 2, radius, 0, Math.PI * 2);
      context.strokeStyle = '#b8ff65';
      context.lineWidth = 4 * scale;
      context.stroke();
    }
  });
  clearTimeout(animationTimer);
  $('scanStage').classList.remove('scanning');
  if (animate && $('scanEffect').checked && !matchMedia('(prefers-reduced-motion: reduce)').matches) {
    requestAnimationFrame(() => {
      $('scanStage').classList.add('scanning');
      animationTimer = setTimeout(() => $('scanStage').classList.remove('scanning'), 1550);
    });
  }
}

function selectPrediction(index, toggle = true) {
  if (!state.result) return;
  state.selected = toggle && state.selected === index ? -1 : index;
  $('regionSelect').value = String(state.selected);
  $('clearRegion').disabled = state.selected < 0;
  const prediction = state.result.predictions[state.selected];
  $('selectedLabel').textContent = prediction ? `Lỗi ${state.selected + 1} · ${prediction.class_vi} · ${formatPercent(prediction.score)}` : '';
  [...$('predictionTable').tBodies[0].rows].forEach((row, rowIndex) => {
    row.classList.toggle('selected', rowIndex === state.selected);
    if (rowIndex === state.selected) row.setAttribute('aria-current', 'true');
    else row.removeAttribute('aria-current');
  });
  drawResult();
}

$('regionSelect').addEventListener('change', () => selectPrediction(Number($('regionSelect').value), false));
$('clearRegion').addEventListener('click', () => selectPrediction(-1, false));
$('resultCanvas').addEventListener('click', (event) => {
  if (!state.result) return;
  const canvas = $('resultCanvas');
  const bounds = canvas.getBoundingClientRect();
  const x = (event.clientX - bounds.left) * canvas.width / bounds.width;
  const y = (event.clientY - bounds.top) * canvas.height / bounds.height;
  const padding = 5 * canvas.width / bounds.width;
  // Prefer the smallest matching box when predicted regions overlap.
  const hits = state.result.predictions.map((prediction, index) => ({ ...prediction, index }))
    .filter(({ bbox_xyxy: [x1, y1, x2, y2] }) =>
      x >= x1 - padding && x <= x2 + padding && y >= y1 - padding && y <= y2 + padding)
    .sort((a, b) => (a.bbox_xyxy[2] - a.bbox_xyxy[0]) * (a.bbox_xyxy[3] - a.bbox_xyxy[1])
      - (b.bbox_xyxy[2] - b.bbox_xyxy[0]) * (b.bbox_xyxy[3] - b.bbox_xyxy[1]));
  selectPrediction(hits[0]?.index ?? -1);
});

function renderPredictions() {
  const result = state.result;
  const predictions = result.predictions;
  $('regionSelect').replaceChildren(new Option('Hiển thị tất cả vùng lỗi', '-1'),
    ...predictions.map((prediction, index) => new Option(
      `Lỗi ${index + 1} · ${prediction.class_vi} · ${formatPercent(prediction.score)}`, String(index))));
  $('regionControls').hidden = predictions.length === 0;
  $('clearRegion').disabled = true;
  const mean = predictions.length ? predictions.reduce((sum, p) => sum + p.score, 0) / predictions.length : null;
  $('predictionMetrics').replaceChildren(
    metric('Vùng lỗi được dự đoán', predictions.length),
    metric('Loại lỗi xuất hiện', new Set(predictions.map((p) => p.class_id)).size),
    metric('Tin cậy trung bình', mean == null ? '—' : `${(mean * 100).toFixed(1)}%`),
    metric('Thời gian xử lý CPU', `${Math.round(result.processing_ms)} ms`)
  );
  $('predictionCount').textContent = `${predictions.length} vùng dự đoán`;
  $('tileCount').textContent = result.tiles_scanned > 1 ? `${result.tiles_scanned} vùng ảnh đã quét` : '';
  const tbody = $('predictionTable').tBodies[0];
  tbody.replaceChildren();
  predictions.forEach((prediction, index) => {
    const row = element('tr', undefined, 'prediction-row');
    row.tabIndex = 0;
    row.setAttribute('aria-label', `Khoanh vị trí lỗi ${index + 1}, ${prediction.class_vi}`);
    row.append(element('td', index + 1));
    const label = element('td');
    const cell = element('span', undefined, 'class-cell');
    const dot = element('i', undefined, 'color-dot');
    dot.style.setProperty('--color', colors[prediction.class_id]);
    cell.append(dot, document.createTextNode(prediction.class_vi));
    label.append(cell);
    row.append(label, element('td', formatPercent(prediction.score), 'confidence-cell'),
      element('td', prediction.bbox_xyxy.map((v) => v.toFixed(1)).join(' · '), 'mono'));
    row.addEventListener('click', () => selectPrediction(index));
    row.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        selectPrediction(index);
      }
    });
    tbody.append(row);
  });
  $('emptyResult').hidden = predictions.length > 0;
  $('predictionTable').parentElement.hidden = predictions.length === 0;
  $('review').hidden = false;
}

async function loadSample() {
  const sample = state.samples.find((value) => value.id === $('sampleSelect').value);
  if (!sample) return;
  cancelPrediction();
  state.file = null;
  state.result = null;
  $('review').hidden = true;
  $('imageGrid').hidden = true;
  const version = ++state.imageVersion;
  try {
    const response = await fetch(sample.url);
    if (!response.ok) throw new Error('Không tải được ảnh mẫu. Hãy thử lại.');
    const blob = await response.blob();
    if (version !== state.imageVersion) return;
    await loadFile(new File([blob], `${sample.id}.jpg`, { type: 'image/jpeg' }));
  } catch (error) {
    if (version === state.imageVersion) notice(error.message, true);
  }
}

function fillSamples() {
  const samples = state.samples.filter((sample) => sample.split === $('sampleSplit').value);
  $('sampleSelect').replaceChildren(...samples.map((sample) =>
    new Option(`${sample.id} · ${sample.labels} lỗi trong nhãn gốc`, sample.id)));
  loadSample();
}

document.querySelectorAll('input[name=source]').forEach((radio) => {
  radio.addEventListener('change', () => {
    const sampleMode = radio.value === 'sample';
    $('sampleControls').hidden = !sampleMode;
    $('uploadControls').hidden = sampleMode;
    state.imageVersion += 1;
    cancelPrediction();
    if (sampleMode) loadSample();
    else if ($('imageFile').files[0]) loadFile($('imageFile').files[0]);
    else {
      state.file = null;
      state.result = null;
      $('imageGrid').hidden = true;
      $('review').hidden = true;
      $('imageDimensions').textContent = '';
      notice('Chọn một ảnh PCB để bắt đầu phân tích.');
    }
  });
});
$('sampleSplit').addEventListener('change', fillSamples);
$('sampleSelect').addEventListener('change', loadSample);
$('imageFile').addEventListener('change', () => {
  const file = $('imageFile').files[0];
  $('uploadName').textContent = file ? file.name : 'PNG, JPEG, BMP, TIFF · Tối đa 4 MB';
  loadFile(file);
});
['dragenter', 'dragover'].forEach((event) => $('uploadControls').addEventListener(event, () => $('uploadControls').classList.add('drag-over')));
['dragleave', 'drop'].forEach((event) => $('uploadControls').addEventListener(event, () => $('uploadControls').classList.remove('drag-over')));
$('confidence').addEventListener('input', () => {
  $('confidenceValue').value = Number($('confidence').value).toFixed(2);
  cancelPrediction();
  state.result = null;
  $('review').hidden = true;
  thresholdTimer = setTimeout(analyze, 250);
});
$('enhance').addEventListener('change', analyze);
$('scanEffect').addEventListener('change', () => {
  $('downloadImage').textContent = $('scanEffect').checked ? 'Tải ảnh kết quả GIF' : 'Tải ảnh kết quả PNG';
  if (!$('scanEffect').checked) {
    clearTimeout(animationTimer);
    $('scanStage').classList.remove('scanning');
  }
});

function download(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = element('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1500);
}

$('downloadJson').addEventListener('click', () => {
  if (!state.result) return;
  download(new Blob([JSON.stringify({ ...state.result, filename: state.file.name }, null, 2)], { type: 'application/json' }), 'pcb_prediction.json');
});
$('downloadImage').addEventListener('click', async () => {
  if (!state.result) return;
  const button = $('downloadImage');
  button.disabled = true;
  button.textContent = 'Đang tạo ảnh…';
  const animation = $('scanEffect').checked;
  const body = new FormData();
  body.append('file', state.file);
  body.append('predictions', JSON.stringify(state.result.predictions));
  body.append('selected', String(state.selected));
  body.append('animation', String(animation));
  try {
    const response = await fetch('/api/export', { method: 'POST', body });
    if (!response.ok) await responseJson(response);
    download(await response.blob(), `pcb_result.${animation ? 'gif' : 'png'}`);
  } catch (error) {
    notice(error.message || 'Chưa tải được ảnh kết quả.', true);
  } finally {
    button.disabled = false;
    button.textContent = $('scanEffect').checked ? 'Tải ảnh kết quả GIF' : 'Tải ảnh kết quả PNG';
  }
});

function experimentName(experiment) {
  return `${experiment.model.toUpperCase()} · ${modes[experiment.mode]}`;
}

function renderExperiments() {
  const split = $('experimentSplit').value;
  const checked = new Set([...document.querySelectorAll('.experiment-choices input:checked')].map((input) => input.value));
  const tbody = $('experimentTable').tBodies[0];
  const chart = $('experimentChart');
  tbody.replaceChildren();
  chart.replaceChildren();
  state.experiments.filter((e) => checked.has(e.run)).forEach((experiment) => {
    const m = experiment.metrics[split];
    if (!m) return;
    const row = element('tr');
    addCells(row, [experimentName(experiment), ...['precision', 'recall', 'f1', 'map50', 'map50_95'].map((key) => formatPercent(m[key]))]);
    tbody.append(row);
    const bar = element('div', undefined, 'bar-row');
    const track = element('div', undefined, 'bar-track');
    const fill = element('div', undefined, 'bar-fill');
    fill.style.width = `${m.map50_95 * 100}%`;
    track.append(fill);
    bar.append(element('span', experimentName(experiment)), track, element('strong', formatPercent(m.map50_95)));
    chart.append(bar);
  });
  if (!tbody.children.length) chart.append(element('p', 'Chọn ít nhất một phương án để xem kết quả.', 'muted'));
  renderDetail();
}

function renderDetail() {
  const experiment = state.experiments.find((e) => e.run === $('experimentDetail').value);
  const m = experiment?.metrics[$('experimentSplit').value];
  $('experimentMetrics').replaceChildren();
  $('classTable').tBodies[0].replaceChildren();
  if (!m) return;
  $('experimentMetrics').append(...['precision', 'recall', 'f1', 'map50_95'].map((key, index) =>
    metric(['Precision', 'Recall', 'F1-score', 'mAP@50–95'][index], formatPercent(m[key]))));
  m.per_class.forEach((c) => {
    const row = element('tr');
    addCells(row, [vietnamese[names.indexOf(c.name)] || c.name, c.ground_truth, c.tp, c.fp, c.fn, formatPercent(c.ap50_95)]);
    $('classTable').tBodies[0].append(row);
  });
}

$('experimentSplit').addEventListener('change', renderExperiments);
$('experimentDetail').addEventListener('change', renderDetail);

async function initialize() {
  try {
    const [samples, experiments, audit] = await Promise.all(
      ['/static/samples.json', '/static/experiments.json', '/static/audit.json']
        .map(async (url) => responseJson(await fetch(url)))
    );
    state.samples = samples;
    state.experiments = experiments;
    experiments.forEach((experiment) => {
      const label = element('label');
      const input = element('input');
      input.type = 'checkbox';
      input.value = experiment.run;
      input.checked = true;
      input.addEventListener('change', renderExperiments);
      label.append(input, document.createTextNode(experimentName(experiment)));
      $('experimentChoices').append(label);
      $('experimentDetail').append(new Option(experimentName(experiment), experiment.run));
    });
    $('experimentDetail').value = 'yolov8n_pair_full_01';
    renderExperiments();
    const counts = Object.values(audit.splits);
    const pairs = counts.reduce((sum, c) => sum + c.pairs, 0);
    const boxes = counts.reduce((sum, c) => sum + c.boxes, 0);
    $('datasetMetrics').append(metric('Cặp ảnh', pairs.toLocaleString()), metric('Tổng ảnh', (pairs * 2).toLocaleString()), metric('Hộp nhãn', boxes.toLocaleString()), metric('Loại lỗi', 6));
    Object.entries(audit.splits).forEach(([split, values]) => {
      const row = element('tr');
      addCells(row, [splits[split], values.pairs, values.boxes, { train: 'Học trọng số', val: 'Lựa chọn mô hình', test: 'Đánh giá cuối' }[split]]);
      $('datasetTable').tBodies[0].append(row);
    });
    fillSamples();
  } catch (error) {
    notice(`Chưa tải được dữ liệu ứng dụng. ${error.message}`, true);
  }
}

initialize();
