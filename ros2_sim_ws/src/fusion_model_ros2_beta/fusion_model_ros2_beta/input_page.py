"""Small standard-library HTTP page for selecting a character to write."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlsplit


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ROS 2 毛笔字书写 Beta</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { max-width: 860px; margin: 32px auto; padding: 0 18px; color: #202124; }
    h1 { margin-bottom: 8px; }
    .sub { color: #5f6368; margin-top: 0; }
    .card { border: 1px solid #dadce0; border-radius: 12px; padding: 18px; margin: 18px 0; }
    .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; }
    input { font-size: 32px; width: 7em; text-align: center; padding: 8px; border: 1px solid #9aa0a6; border-radius: 8px; }
    input.dimension { font-size: 18px; width: 5.5em; padding: 10px; }
    label { color: #5f6368; }
    select { font-size: 18px; padding: 10px; margin-left: 6px; border: 1px solid #9aa0a6; border-radius: 8px; background: white; }
    button { font-size: 18px; padding: 10px 18px; border: 0; border-radius: 8px; background: #1a73e8; color: white; cursor: pointer; }
    button.secondary { background: #5f6368; }
    button:disabled { background: #b7b7b7; cursor: wait; }
    #message { white-space: pre-wrap; line-height: 1.55; margin-top: 14px; }
    .ok { color: #137333; } .warn { color: #b06000; } .bad { color: #c5221f; }
    table { border-collapse: collapse; width: 100%; margin-top: 10px; }
    td { padding: 6px 4px; border-bottom: 1px solid #eee; }
    td:first-child { color: #5f6368; width: 10em; }
    code { word-break: break-all; }
  </style>
</head>
<body>
  <h1>ROS 2 毛笔字书写 Beta</h1>
  <p class="sub">一次输入 1～3 个楷书字。设置书写区域长宽，按排版、字数和间距自动计算各字大小，再执行姿态反演。纸面和支撑台尺寸不变。</p>
  <div class="card">
    <div class="controls">
      <input id="character" maxlength="3" autocomplete="off" autofocus aria-label="输入一个或多个汉字" placeholder="例如：武字">
      <label for="layout">排版
        <select id="layout">
          <option value="horizontal">横向（从左到右）</option>
          <option value="vertical">竖向（从上到下）</option>
        </select>
      </label>
      <label for="writing-width">书写区域宽
        <input class="dimension" id="writing-width" type="number" min="1" max="800" step="5" value="520"> mm
      </label>
      <label for="writing-height">书写区域高
        <input class="dimension" id="writing-height" type="number" min="1" max="400" step="5" value="320"> mm
      </label>
      <button id="write">开始写字</button>
      <button id="clear" class="secondary">清除字迹</button>
      <button id="refresh" class="secondary">刷新状态</button>
    </div>
    <div id="message">正在读取轨迹目录……</div>
  </div>
  <div class="card">
    <strong>当前任务</strong>
    <table id="status"></table>
  </div>
  <div class="card">
    <strong>轨迹目录</strong>
    <table id="catalog"></table>
  </div>
<script>
const $ = id => document.getElementById(id);
let dimensionsInitialized = false;
function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function setMessage(text, cls='') { $('message').className = cls; $('message').textContent = text; }
async function getJson(url, options) {
  const r = await fetch(url, options);
  const body = await r.json();
  if (!r.ok) throw new Error(body.message || body.error || ('HTTP ' + r.status));
  return body;
}
async function refresh() {
  try {
    const q = $('character').value.trim();
    const [s, c] = await Promise.all([getJson('/api/status'), getJson('/api/catalog?q=' + encodeURIComponent(q))]);
    const i = s.inversion_interface || {};
    if (!dimensionsInitialized) {
      $('writing-width').value = Number(s.writing_width_m || .52)*1000;
      $('writing-height').value = Number(s.writing_height_m || .32)*1000;
      $('writing-width').max = Number(s.paper_width_m || .8)*1000;
      $('writing-height').max = Number(s.paper_height_m || .4)*1000;
      dimensionsInitialized = true;
    }
    $('status').innerHTML = '<tr><td>状态</td><td>' + esc(s.status) + '</td></tr>'
      + '<tr><td>文字</td><td>' + esc(s.character || '—') + '</td></tr>'
      + '<tr><td>排版</td><td>' + esc(s.layout_label || '—') + '</td></tr>'
      + '<tr><td>自动计算字号</td><td>' + esc((s.computed_font_sizes_mm || [Number(s.font_size_m || 0)*1000]).map(v => Number(v).toFixed(1)).join(' / ') + ' mm') + '</td></tr>'
      + '<tr><td>书写区域</td><td>' + esc(((Number(s.writing_width_m || 0) * 1000).toFixed(0)) + ' × ' + ((Number(s.writing_height_m || 0) * 1000).toFixed(0)) + ' mm') + '</td></tr>'
      + '<tr><td>模型权重</td><td>' + esc(i.model_weight_version || '—') + '</td></tr>'
      + '<tr><td>反演流程</td><td><code>' + esc(i.inversion_pipeline || '—') + '</code></td></tr>'
      + '<tr><td>权重 SHA-256</td><td><code>' + esc(i.checkpoint_sha256 || '—') + '</code></td></tr>'
      + '<tr><td>字段接口</td><td>' + esc('固定 ' + (i.fixed_fields || []).join('/') + '；优化 ' + (i.optimized_fields || []).join('/') + '；推导 ' + (i.derived_fields || []).join('/')) + '</td></tr>'
      + '<tr><td>进度</td><td>' + esc((Number(s.progress || 0) * 100).toFixed(1)) + '%</td></tr>'
      + '<tr><td>字迹</td><td>' + (s.ink_present ? '已显示（请先清除）' : '已清除') + '</td></tr>'
      + '<tr><td>消息</td><td>' + esc(s.message || '—') + '</td></tr>';
    $('catalog').innerHTML = '<tr><td>已完成</td><td>' + esc(c.ready_count) + '</td></tr>'
      + '<tr><td>生成中</td><td>' + esc(c.generating_count) + '</td></tr>'
      + '<tr><td>不可用/失败</td><td>' + esc(c.unavailable_count) + '</td></tr>'
      + '<tr><td>查询结果</td><td>' + (c.entries.length ? c.entries.map(e => esc(e.character) + '：' + esc(e.status)).join('<br>') : '无') + '</td></tr>';
    const busy = ['offline_generating', 'queued', 'preparing', 'running'].includes(s.status);
    $('write').disabled = busy || Boolean(s.ink_present);
    // Keep the button available when idle so a stale RViz marker can be
    // cleared again even if the ROS node has already reset its local cache.
    $('clear').disabled = busy;
    if (q && c.entries.length === 1 && c.entries[0].ready) setMessage('轨迹已就绪，可以开始写字。', 'ok');
  } catch (e) { setMessage('读取状态失败：' + e.message, 'bad'); }
}
async function writeCharacter() {
  const character = $('character').value.trim();
  const layout = $('layout').value;
  const writing_width_mm = Number($('writing-width').value);
  const writing_height_mm = Number($('writing-height').value);
  if (![writing_width_mm, writing_height_mm].every(v => Number.isFinite(v) && v > 0)) {
    setMessage('书写区域长宽必须大于零。', 'warn'); return;
  }
  const count = [...character].filter(value => !/\s/.test(value)).length;
  if (count < 1 || count > 3) { setMessage('请输入 1～3 个字。', 'warn'); return; }
  $('write').disabled = true;
  try {
    const body = await getJson('/api/write', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({character, layout, writing_width_mm, writing_height_mm})});
    setMessage(body.message || '已开始写字。', 'ok');
    await refresh();
  } catch (e) { setMessage(e.message, 'bad'); await refresh(); }
  finally { await refresh(); }
}
async function clearInk() {
  $('clear').disabled = true;
  try {
    const body = await getJson('/api/clear', {method:'POST'});
    setMessage(body.message || '字迹已清除。', 'ok');
  } catch (e) { setMessage(e.message, 'bad'); }
  finally { await refresh(); }
}
$('write').addEventListener('click', writeCharacter);
$('clear').addEventListener('click', clearInk);
$('refresh').addEventListener('click', refresh);
$('character').addEventListener('input', refresh);
$('character').addEventListener('keydown', e => { if (e.key === 'Enter') writeCharacter(); });
refresh();
setInterval(refresh, 1000);
</script>
</body>
</html>
"""


class _InputPageHandler(BaseHTTPRequestHandler):
    """HTTP adapter; all ROS mutations are queued onto the ROS executor."""

    server_version = "FusionModelROS2Input/1.0"

    @property
    def controller(self):
        return self.server.controller  # type: ignore[attr-defined]

    def _respond(self, status: int, payload: Any, content_type: str = "application/json") -> None:
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            self._respond(200, PAGE, "text/html")
            return
        if parsed.path == "/api/status":
            self._respond(200, self.controller.input_status())
            return
        if parsed.path == "/api/catalog":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self._respond(200, self.controller.catalog_summary(query))
            return
        self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlsplit(self.path)
        if parsed.path == "/api/clear":
            result = self.controller.clear_current_ink()
            self._respond(200 if result.get("accepted") else 409, result)
            return
        if parsed.path != "/api/write":
            self._respond(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            character = str(payload.get("character", "")).strip()
            layout = str(payload.get("layout", "horizontal")).strip()
            font_size_m = float(payload['font_size_mm']) / 1000.0 if 'font_size_mm' in payload else None
            width = float(payload['writing_width_mm']) / 1000.0 if 'writing_width_mm' in payload else None
            height = float(payload['writing_height_mm']) / 1000.0 if 'writing_height_mm' in payload else None
        except (ValueError, TypeError, UnicodeError, json.JSONDecodeError) as error:
            self._respond(400, {"error": "bad request", "message": str(error)})
            return
        result = self.controller.request_character(character, layout, font_size_m,
                                                   writing_width_m=width, writing_height_m=height)
        self._respond(202 if result.get("accepted") else 409, result)

    def log_message(self, format: str, *args: Any) -> None:
        # ROS logs are the single source of runtime diagnostics.
        self.controller.get_logger().debug("input page: " + format % args)


class CharacterInputPage:
    """Run the local input page on a daemon thread."""

    def __init__(self, controller: Any, host: str, port: int) -> None:
        self.controller = controller
        self.host = host
        self.port = int(port)
        self.server = ThreadingHTTPServer((self.host, self.port), _InputPageHandler)
        self.server.controller = controller  # type: ignore[attr-defined]
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def address(self) -> Tuple[str, int]:
        host, port = self.server.server_address[:2]
        return str(host), int(port)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
