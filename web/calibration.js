'use strict';

const cameraIndex = location.pathname.split('/').pop();
const endpoint = `/api/calibration/${cameraIndex}`;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');
const saveButton = document.getElementById('save');
const instructions = {
  roi: 'ROI：沿目标道路边缘依次点击，点击起点或“完成区域”闭合。',
  line: '计数线：点击两个端点，横跨目标道路，整段位于 ROI 内。',
  direction: '通行方向：先点来车一侧，再点驶离一侧，画出行驶方向箭头。',
};
let geometry = {roi: [], line: [], direction: [], closed: false};
let mode = 'roi';
let background = null;
let history = [];
let baseline = JSON.stringify(geometry);
let loaded = false;
let saving = false;

function message(text, error = false) {
  status.textContent = text;
  status.dataset.error = String(error);
}

function changed() {
  return JSON.stringify(geometry) !== baseline;
}

function remember() {
  history.push(JSON.stringify(geometry));
}

function update() {
  document.querySelectorAll('[data-mode]').forEach(button => {
    button.setAttribute('aria-pressed', String(button.dataset.mode === mode));
    button.disabled = !loaded || saving;
  });
  document.getElementById('instruction').textContent = instructions[mode];
  document.getElementById('close-roi').disabled = !loaded || saving || geometry.closed || geometry.roi.length < 3;
  document.getElementById('undo').disabled = !loaded || saving || !history.length;
  document.getElementById('clear').disabled = !loaded || saving;
  document.getElementById('refresh').disabled = !loaded || saving;
  saveButton.disabled = !loaded || saving;
  document.getElementById('saved-state').textContent = changed() ? '有未保存的修改' : '当前标定已与保存内容同步';
  draw();
}

function drawPath(points, color, close, label) {
  if (!points.length) return;
  const scale = canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);
  ctx.beginPath();
  points.forEach(([x, y], i) => ctx[i ? 'lineTo' : 'moveTo'](x * canvas.width, y * canvas.height));
  if (close) {
    ctx.closePath();
    ctx.fillStyle = '#34d39926';
    ctx.fill();
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5 * scale;
  ctx.stroke();
  points.forEach(([x, y], i) => {
    ctx.beginPath();
    ctx.arc(x * canvas.width, y * canvas.height, 4 * scale, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.font = `bold ${13 * scale}px system-ui`;
    ctx.fillText(label === 'ROI' ? String(i + 1) : `${label}${i + 1}`, x * canvas.width + 7 * scale, y * canvas.height - 7 * scale);
  });
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (background) ctx.drawImage(background, 0, 0, canvas.width, canvas.height);
  drawPath(geometry.roi, '#6ee7b7', geometry.closed, 'ROI');
  drawPath(geometry.line, '#fbbf24', false, 'L');
  drawPath(geometry.direction, '#67e8f9', false, 'D');
  if (geometry.direction.length === 2) {
    const [a, b] = geometry.direction.map(([x, y]) => [x * canvas.width, y * canvas.height]);
    const angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const length = 16 * canvas.width / Math.max(canvas.getBoundingClientRect().width, 1);
    ctx.beginPath();
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - length * Math.cos(angle - Math.PI / 6), b[1] - length * Math.sin(angle - Math.PI / 6));
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(b[0] - length * Math.cos(angle + Math.PI / 6), b[1] - length * Math.sin(angle + Math.PI / 6));
    ctx.stroke();
  }
}

function closeROI() {
  if (!loaded || saving || geometry.closed || geometry.roi.length < 3) return;
  remember();
  geometry.closed = true;
  mode = 'line';
  message('ROI 已闭合。接下来点击两个端点绘制计数线，或直接保存 ROI。');
  update();
}

canvas.addEventListener('pointerdown', event => {
  if (!loaded || saving || event.button !== 0) return;
  const rect = canvas.getBoundingClientRect();
  const point = [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
                 Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))];
  if (mode !== 'roi' && !geometry.closed) {
    message('请先完成 ROI 区域。', true);
    return;
  }
  if (mode === 'direction' && geometry.line.length !== 2) {
    message('请先画好计数线。', true);
    return;
  }
  if (mode === 'roi' && !geometry.closed && geometry.roi.length >= 3) {
    const first = geometry.roi[0];
    if (Math.hypot((point[0] - first[0]) * rect.width, (point[1] - first[1]) * rect.height) < 10) {
      closeROI();
      return;
    }
  }
  if (mode === 'roi' && !geometry.closed && geometry.roi.length >= 64) {
    message('ROI 最多允许 64 个点，请闭合或撤销。', true);
    return;
  }
  remember();
  if (mode === 'roi' && geometry.closed) geometry = {roi: [], line: [], direction: [], closed: false};
  if (mode !== 'roi' && geometry[mode].length === 2) geometry[mode] = [];
  if (mode === 'line') geometry.direction = [];
  geometry[mode].push(point);
  if (mode === 'line' && geometry.line.length === 2) mode = 'direction';
  message(instructions[mode]);
  update();
});

document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click', () => {
  mode = button.dataset.mode;
  message(instructions[mode]);
  update();
}));
document.getElementById('close-roi').addEventListener('click', closeROI);
document.getElementById('undo').addEventListener('click', () => {
  if (saving || !history.length) return;
  geometry = JSON.parse(history.pop());
  mode = !geometry.closed ? 'roi' : geometry.line.length !== 2 ? 'line' : 'direction';
  message('已撤销上一步。');
  update();
});
document.getElementById('clear').addEventListener('click', () => {
  remember();
  geometry = {roi: [], line: [], direction: [], closed: false};
  mode = 'roi';
  message('已清空。点击保存可移除已保存的标定，也可以重新绘制。');
  update();
});

async function snapshot() {
  const response = await fetch(`/calibration-frame/${cameraIndex}`, {cache: 'no-store'});
  if (!response.ok) throw new Error('尚无有效视频帧，请确认视频已连接后重试。');
  const url = URL.createObjectURL(await response.blob());
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

function setBackground(image) {
  background = image;
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
}

document.getElementById('refresh').addEventListener('click', async () => {
  const button = document.getElementById('refresh');
  button.disabled = true;
  try {
    const image = await snapshot();
    if (Math.abs(image.naturalWidth / image.naturalHeight / (canvas.width / canvas.height) - 1) > 0.002) {
      message('画面比例已变化，请重新打开本标定页并重新绘制。', true);
      return;
    }
    setBackground(image);
    message('已刷新冻结底图，图形保持不变。');
    draw();
  } catch (error) {
    message(error.message, true);
  } finally {
    button.disabled = false;
  }
});

saveButton.addEventListener('click', async () => {
  if (!loaded || saving) return;
  if (geometry.roi.length && !geometry.closed) {
    message('请先点击起点或“完成区域”闭合 ROI。', true);
    return;
  }
  saving = true;
  update();
  try {
    const {roi, line, direction} = geometry;
    const response = await fetch(endpoint, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({roi, line, direction, frame_size: [canvas.width, canvas.height]})});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || '保存失败');
    baseline = JSON.stringify(geometry);
    message('标定已保存。返回监控预览即可查看叠加图形。');
  } catch (error) {
    message(error.message, true);
  } finally {
    saving = false;
    update();
  }
});

window.addEventListener('beforeunload', event => {
  if (changed()) {event.preventDefault(); event.returnValue = '';}
});
window.addEventListener('resize', draw);
document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'z') {
    event.preventDefault();
    document.getElementById('undo').click();
  }
});

async function load() {
  try {
    const response = await fetch(endpoint, {cache: 'no-store'});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || '无法加载标定');
    document.getElementById('title').textContent = `${data.camera} · 区域标定`;
    setBackground(await snapshot());
    let mismatch = false;
    if (data.calibration) {
      const saved = data.calibration;
      mismatch = Math.abs(saved.frame_size[0] / saved.frame_size[1] / (canvas.width / canvas.height) - 1) > 0.002;
      if (!mismatch) geometry = {roi: saved.roi, line: saved.line, direction: saved.direction, closed: saved.roi.length >= 3};
    }
    baseline = JSON.stringify(geometry);
    loaded = true;
    message(mismatch ? '画面比例与原标定不一致，旧标定已暂停叠加，请重新绘制并保存。' : '底图已冻结。选择工具并点击画面开始标定。', mismatch);
    update();
  } catch (error) {
    message(`${error.message} 请刷新页面重试。`, true);
  }
}
load();
