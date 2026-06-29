"""
CordBaro Flask Web UI - Live camera + defect popup

- Title: CordBaro
- Subtitle: 듀얼 로봇 기반 바코드 검출 시스템
- Subscribe /sorter_switch
    0 = OK / normal box
    1 = FAIL / defect box
- Subscribe camera image topic, default /rgb_L
- Show live process video using MJPEG streaming
- When a defect is detected, capture the latest RGB frame and show a popup in the browser

Run example:
    source /opt/ros/humble/setup.bash
    source ~/cobot3_ws/install/setup.bash
    python3 cordbaro_flask_live_ui_popup.py --host 0.0.0.0 --port 8000 --image-topic /rgb_L

Compressed topic example:
    python3 cordbaro_flask_live_ui_popup.py --image-topic /rgb_L/compressed --compressed
"""

import argparse
import datetime as _dt
import threading
import time
from collections import deque

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from sensor_msgs.msg import Image, CompressedImage


HTML = r"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CordBaro</title>
  <style>
    :root {
      --bg: #edf6fb;
      --card: rgba(255,255,255,0.92);
      --line: #dbeaf2;
      --navy: #102a43;
      --muted: #6b7c8f;
      --cyan: #15c8c8;
      --cyan-dark: #0b9eab;
      --ok: #16a34a;
      --fail: #ef4444;
      --warning: #f59e0b;
      --shadow: 0 18px 40px rgba(15, 42, 67, 0.13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR", Arial, sans-serif;
      color: var(--navy);
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(21, 200, 200, 0.22), transparent 34%),
        linear-gradient(135deg, #f8fcff 0%, var(--bg) 100%);
    }
    .wrap { max-width: 1320px; margin: 0 auto; padding: 34px 28px 42px; }
    header {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 24px;
    }
    .brand h1 {
      margin: 0;
      font-size: 58px;
      line-height: 1;
      letter-spacing: -1.8px;
      color: #082f49;
    }
    .brand p {
      margin: 10px 0 0;
      color: var(--muted);
      font-size: 20px;
      font-weight: 650;
      letter-spacing: -0.4px;
    }
    .status-pill {
      min-width: 210px;
      border-radius: 999px;
      padding: 16px 24px;
      text-align: center;
      font-size: 22px;
      font-weight: 900;
      letter-spacing: 0.4px;
      color: white;
      box-shadow: var(--shadow);
      background: var(--muted);
    }
    .status-ok { background: linear-gradient(135deg, #22c55e, #15803d); }
    .status-fail { background: linear-gradient(135deg, #fb7185, #dc2626); animation: pulse 1.2s infinite; }
    .status-wait { background: linear-gradient(135deg, #64748b, #334155); }
    @keyframes pulse { 0%,100%{ transform: scale(1); } 50%{ transform: scale(1.03); } }

    .stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 18px;
      margin-bottom: 22px;
    }
    .stat-card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 22px 24px;
      box-shadow: var(--shadow);
    }
    .stat-label { color: var(--muted); font-size: 16px; font-weight: 750; }
    .stat-value { margin-top: 8px; font-size: 44px; line-height: 1; font-weight: 950; letter-spacing: -1px; }
    .ok-text { color: var(--ok); }
    .fail-text { color: var(--fail); }

    .main-grid {
      display: grid;
      grid-template-columns: 1.2fr 0.8fr;
      gap: 22px;
      align-items: stretch;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 28px;
      padding: 22px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel h2 {
      margin: 0 0 16px;
      font-size: 22px;
      letter-spacing: -0.5px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .small-tag {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 850;
      color: #075985;
      background: #dff7fb;
      border: 1px solid #b8edf4;
    }
    .live-box, .defect-box {
      width: 100%;
      background: #d8e9f2;
      border-radius: 22px;
      overflow: hidden;
      border: 1px solid #c9dde8;
      position: relative;
    }
    .live-box { aspect-ratio: 16 / 9; }
    .defect-box { aspect-ratio: 16 / 10; }
    .live-box img, .defect-box img {
      display: block;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
    .placeholder {
      position: absolute;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-weight: 800;
      text-align: center;
      padding: 24px;
    }
    .side-stack { display: grid; gap: 22px; }
    .log-list { display: grid; gap: 10px; max-height: 300px; overflow-y: auto; padding-right: 4px; }
    .log-row {
      display: grid;
      grid-template-columns: 92px 70px 1fr;
      align-items: center;
      gap: 10px;
      padding: 12px 14px;
      background: rgba(239, 246, 251, 0.75);
      border: 1px solid #dcecf5;
      border-radius: 16px;
      font-size: 14px;
    }
    .badge {
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 950;
      text-align: center;
      color: white;
    }
    .badge.ok { background: var(--ok); }
    .badge.fail { background: var(--fail); }
    .empty-log { color: var(--muted); padding: 22px 10px; text-align: center; font-weight: 700; }

    .actions { display: flex; gap: 10px; margin-top: 16px; }
    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font-weight: 900;
      cursor: pointer;
      color: #083344;
      background: #dff7fb;
      border: 1px solid #b8edf4;
    }
    button:hover { filter: brightness(0.98); }

    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(2, 6, 23, 0.55);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 1000;
      padding: 24px;
    }
    .modal-backdrop.show { display: flex; }
    .modal {
      width: min(920px, 96vw);
      background: white;
      border-radius: 30px;
      overflow: hidden;
      box-shadow: 0 30px 90px rgba(0,0,0,0.35);
      border: 1px solid rgba(255,255,255,0.5);
    }
    .modal-head {
      background: linear-gradient(135deg, #ef4444, #be123c);
      color: white;
      padding: 22px 26px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
    }
    .modal-head h3 { margin: 0; font-size: 28px; letter-spacing: -0.6px; }
    .modal-head p { margin: 6px 0 0; opacity: 0.9; font-weight: 650; }
    .close-btn {
      background: rgba(255,255,255,0.18);
      color: white;
      border: 1px solid rgba(255,255,255,0.25);
      font-size: 16px;
      flex-shrink: 0;
    }
    .modal-body {
      padding: 24px;
      display: grid;
      grid-template-columns: 1fr 0.62fr;
      gap: 20px;
      align-items: start;
    }
    .modal-img {
      aspect-ratio: 16/9;
      background: #e2e8f0;
      border-radius: 20px;
      overflow: hidden;
      border: 1px solid #cbd5e1;
    }
    .modal-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .modal-info {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 20px;
      padding: 18px;
    }
    .modal-info b { color: #be123c; }
    .modal-info p { margin: 0 0 14px; color: #334155; line-height: 1.55; font-weight: 650; }

    @media (max-width: 920px) {
      header { align-items: flex-start; flex-direction: column; }
      .brand h1 { font-size: 46px; }
      .stats { grid-template-columns: 1fr; }
      .main-grid { grid-template-columns: 1fr; }
      .modal-body { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="brand">
        <h1>CodeBaro</h1>
        <p>듀얼 로봇 기반 바코드 검출 시스템</p>
      </div>
      <div id="statusPill" class="status-pill status-wait">WAITING</div>
    </header>

    <section class="stats">
      <div class="stat-card">
        <div class="stat-label">전체 검사</div>
        <div id="totalCount" class="stat-value">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">정상 판정</div>
        <div id="okCount" class="stat-value ok-text">0</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">불량 판정</div>
        <div id="failCount" class="stat-value fail-text">0</div>
      </div>
    </section>

    <section class="main-grid">
      <div class="panel">
        <h2>실시간 공정 화면 <span class="small-tag">camera_L</span></h2>
        <div class="live-box">
          <img src="/video_feed" alt="live process video" />
        </div>
        <div class="actions">
          <button onclick="resetState()">화면 초기화</button>
          <button onclick="testPopup()">팝업 테스트</button>
        </div>
      </div>

      <div class="side-stack">
        <div class="panel">
          <h2>불량 판정 이미지 <span class="small-tag">FAIL capture</span></h2>
          <div class="defect-box" id="defectBox">
            <div class="placeholder" id="defectPlaceholder">아직 불량 판정 이미지가 없습니다.</div>
            <img id="defectImage" alt="defect capture" style="display:none;" />
          </div>
        </div>

        <div class="panel">
          <h2>최근 판정 로그</h2>
          <div id="logList" class="log-list">
            <div class="empty-log">아직 판정 로그가 없습니다.</div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <div class="modal-backdrop" id="defectModal">
    <div class="modal">
      <div class="modal-head">
        <div>
          <h3>불량 박스 검출</h3>
          <p id="modalTime">6면 검사 후 바코드가 검출되지 않았습니다.</p>
        </div>
        <button class="close-btn" onclick="closePopup()">확인</button>
      </div>
      <div class="modal-body">
        <div class="modal-img">
          <img id="modalImage" alt="defect popup image" />
        </div>
        <div class="modal-info">
          <p><b>판정 결과:</b> FAIL</p>
          <p>해당 박스는 바코드 미검출 박스로 분류되었습니다.</p>
          <p>최근 RGB 프레임과 판정 로그를 함께 저장하여 작업자가 즉시 확인할 수 있습니다.</p>
          <button onclick="closePopup()">확인 완료</button>
        </div>
      </div>
    </div>
  </div>

<script>
  let lastDefectEventId = 0;
  let initialized = false;

  function statusClass(status) {
    if (status === 'FAIL') return 'status-pill status-fail';
    if (status === 'OK') return 'status-pill status-ok';
    return 'status-pill status-wait';
  }

  async function fetchState() {
    try {
      const res = await fetch('/api/state', {cache: 'no-store'});
      const s = await res.json();

      document.getElementById('totalCount').textContent = s.total;
      document.getElementById('okCount').textContent = s.ok;
      document.getElementById('failCount').textContent = s.fail;

      const pill = document.getElementById('statusPill');
      pill.textContent = s.current_status;
      pill.className = statusClass(s.current_status);

      const defectImage = document.getElementById('defectImage');
      const placeholder = document.getElementById('defectPlaceholder');
      if (s.has_defect_image) {
        defectImage.src = '/defect_image.jpg?event=' + s.defect_event_id + '&t=' + Date.now();
        defectImage.style.display = 'block';
        placeholder.style.display = 'none';
      } else {
        defectImage.style.display = 'none';
        placeholder.style.display = 'flex';
      }

      renderLogs(s.logs || []);

      if (!initialized) {
        lastDefectEventId = s.defect_event_id;
        initialized = true;
      } else if (s.defect_event_id > lastDefectEventId) {
        lastDefectEventId = s.defect_event_id;
        showPopup(s);
      }
    } catch (e) {
      console.log('state fetch failed', e);
    }
  }

  function renderLogs(logs) {
    const box = document.getElementById('logList');
    if (!logs.length) {
      box.innerHTML = '<div class="empty-log">아직 판정 로그가 없습니다.</div>';
      return;
    }
    box.innerHTML = logs.map(row => {
      const cls = row.result === 'FAIL' ? 'fail' : 'ok';
      return `<div class="log-row">
        <div>${row.time}</div>
        <div><span class="badge ${cls}">${row.result}</span></div>
        <div>${row.message}</div>
      </div>`;
    }).join('');
  }

  function showPopup(s) {
    document.getElementById('modalTime').textContent = (s.last_defect_time || '') + '  |  6면 검사 후 바코드가 검출되지 않았습니다.';
    document.getElementById('modalImage').src = '/defect_image.jpg?event=' + s.defect_event_id + '&t=' + Date.now();
    document.getElementById('defectModal').classList.add('show');
  }

  function closePopup() {
    document.getElementById('defectModal').classList.remove('show');
  }

  async function resetState() {
    await fetch('/api/reset', {method: 'POST'});
    lastDefectEventId = 0;
    initialized = false;
    closePopup();
    fetchState();
  }

  async function testPopup() {
    await fetch('/api/test_defect', {method: 'POST'});
    fetchState();
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closePopup();
  });

  setInterval(fetchState, 500);
  fetchState();
</script>
</body>
</html>
"""


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.ok = 0
        self.fail = 0
        self.current_status = "WAITING"
        self.logs = deque(maxlen=12)
        self.latest_jpeg = None
        self.defect_jpeg = None
        self.defect_event_id = 0
        self.last_defect_time = ""
        self.last_frame_time = 0.0

    def add_log(self, result: str, message: str):
        now = _dt.datetime.now().strftime("%H:%M:%S")
        self.logs.appendleft({"time": now, "result": result, "message": message})
        return now


STATE = SharedState()


def encode_placeholder(text="camera_L 영상 대기 중"):
    img = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.rectangle(img, (30, 30), (1250, 690), (200, 220, 230), 3)
    cv2.putText(img, text, (360, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (90, 110, 125), 3, cv2.LINE_AA)
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else b''


PLACEHOLDER_JPEG = encode_placeholder()


def image_msg_to_bgr(msg: Image):
    h, w = msg.height, msg.width
    enc = (msg.encoding or '').lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc in ('rgb8', 'bgr8'):
        img = data.reshape((h, w, 3))
        if enc == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    if enc in ('rgba8', 'bgra8'):
        img = data.reshape((h, w, 4))
        if enc == 'rgba8':
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    if enc in ('mono8', '8uc1'):
        img = data.reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Fallback: try 3-channel image
    if len(data) >= h * w * 3:
        return data[:h*w*3].reshape((h, w, 3))
    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def bgr_to_jpeg(img, quality=82):
    ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return buf.tobytes()


class CordBaroRosNode(Node):
    def __init__(self, image_topic: str, sorter_topic: str, compressed: bool):
        super().__init__('cordbaro_flask_live_ui_popup')
        self.image_topic = image_topic
        self.sorter_topic = sorter_topic
        self.compressed = compressed

        self.create_subscription(Int32, sorter_topic, self.sorter_cb, 10)
        if compressed:
            self.create_subscription(CompressedImage, image_topic, self.compressed_image_cb, 10)
        else:
            self.create_subscription(Image, image_topic, self.image_cb, 10)

        self.get_logger().info(f"CordBaro Web UI started. sorter={sorter_topic}, image={image_topic}, compressed={compressed}")

    def sorter_cb(self, msg: Int32):
        value = int(msg.data)
        with STATE.lock:
            STATE.total += 1
            if value == 1:
                STATE.fail += 1
                STATE.current_status = "FAIL"
                STATE.defect_event_id += 1
                STATE.defect_jpeg = STATE.latest_jpeg if STATE.latest_jpeg is not None else PLACEHOLDER_JPEG
                STATE.last_defect_time = STATE.add_log("FAIL", "바코드 미검출 박스 분류")
            else:
                STATE.ok += 1
                STATE.current_status = "OK"
                STATE.add_log("OK", "정상 박스 통과")

    def image_cb(self, msg: Image):
        try:
            bgr = image_msg_to_bgr(msg)
            jpeg = bgr_to_jpeg(bgr)
            if jpeg:
                with STATE.lock:
                    STATE.latest_jpeg = jpeg
                    STATE.last_frame_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"image conversion failed: {e}")

    def compressed_image_cb(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode returned None")
            jpeg = bgr_to_jpeg(bgr)
            if jpeg:
                with STATE.lock:
                    STATE.latest_jpeg = jpeg
                    STATE.last_frame_time = time.time()
        except Exception as e:
            self.get_logger().warn(f"compressed image conversion failed: {e}")


app = Flask(__name__)


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/api/state')
def api_state():
    with STATE.lock:
        return jsonify({
            'total': STATE.total,
            'ok': STATE.ok,
            'fail': STATE.fail,
            'current_status': STATE.current_status,
            'logs': list(STATE.logs),
            'has_defect_image': STATE.defect_jpeg is not None,
            'defect_event_id': STATE.defect_event_id,
            'last_defect_time': STATE.last_defect_time,
            'has_live_frame': STATE.latest_jpeg is not None,
        })


@app.route('/api/reset', methods=['POST'])
def api_reset():
    with STATE.lock:
        STATE.total = 0
        STATE.ok = 0
        STATE.fail = 0
        STATE.current_status = "WAITING"
        STATE.logs.clear()
        STATE.defect_jpeg = None
        STATE.defect_event_id = 0
        STATE.last_defect_time = ""
    return jsonify({'ok': True})


@app.route('/api/test_defect', methods=['POST'])
def api_test_defect():
    with STATE.lock:
        STATE.total += 1
        STATE.fail += 1
        STATE.current_status = "FAIL"
        STATE.defect_event_id += 1
        STATE.defect_jpeg = STATE.latest_jpeg if STATE.latest_jpeg is not None else PLACEHOLDER_JPEG
        STATE.last_defect_time = STATE.add_log("FAIL", "팝업 테스트 - 불량 박스 검출")
    return jsonify({'ok': True})


@app.route('/defect_image.jpg')
def defect_image():
    with STATE.lock:
        jpeg = STATE.defect_jpeg or PLACEHOLDER_JPEG
    return Response(jpeg, mimetype='image/jpeg')


@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with STATE.lock:
                jpeg = STATE.latest_jpeg or PLACEHOLDER_JPEG
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
            time.sleep(0.05)  # about 20 FPS max
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def spin_ros(args):
    rclpy.init()
    node = CordBaroRosNode(args.image_topic, args.sorter_topic, args.compressed)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='CordBaro Flask live Web UI with defect popup')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--image-topic', default='/recoding_camera_01/recoding_rgb')
    parser.add_argument('--sorter-topic', default='/sorter_switch')
    parser.add_argument('--compressed', action='store_true')
    args = parser.parse_args()

    ros_thread = threading.Thread(target=spin_ros, args=(args,), daemon=True)
    ros_thread.start()

    print(f"[WEB] CordBaro UI: http://{args.host}:{args.port}")
    print("[WEB] Defect popup is enabled when sorter_switch=1")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
