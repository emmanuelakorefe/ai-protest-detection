# app.py — Campus Protest Detection MVP (CPU)
# Features: upload -> detect people + face blur -> annotated MP4 + CSV + chart
# Threshold alerts (≥N people for ≥D seconds) -> SNS email with S3 pre-signed links

import os, uuid, csv, json
from typing import List, Dict, Tuple

from flask import Flask, request, send_from_directory, render_template_string, abort
import boto3
import numpy as np
import cv2
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")  # headless for servers
import matplotlib.pyplot as plt
import mimetypes

# --------------------
# Env / AWS clients
# --------------------
AWS_REGION       = os.getenv("AWS_REGION", "eu-central-1").strip()
SNS_TOPIC_ARN    = os.getenv("SNS_TOPIC_ARN", "").strip()
S3_BUCKET        = os.getenv("S3_BUCKET", "").strip()
URL_EXPIRES_SECS = int(os.getenv("URL_EXPIRES_SECS", "3600"))

sns = boto3.client("sns", region_name=AWS_REGION) if SNS_TOPIC_ARN else None
s3  = boto3.client("s3",  region_name=AWS_REGION) if S3_BUCKET  else None

def publish_sns(subject: str, message: str) -> Tuple[bool, str]:
    if not sns or not SNS_TOPIC_ARN:
        return False, "SNS not configured"
    try:
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def upload_and_sign(local_path: str, run_id: str) -> str:
    """Upload local file to s3://<bucket>/runs/<run_id>/<filename> and return pre-signed URL."""
    if not s3 or not S3_BUCKET:
        return ""
    key = f"runs/{run_id}/{os.path.basename(local_path)}"
    ctype, _ = mimetypes.guess_type(local_path)
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": ctype or "application/octet-stream"})
    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=URL_EXPIRES_SECS,
    )
    return url

# --------------------
# Paths / App config
# --------------------
UPLOAD_DIR = os.path.expanduser("~/uploads")
OUTPUT_DIR = os.path.expanduser("~/outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_MB = 500
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# YOLO model (CPU-friendly nano)
model = YOLO("yolov8n.pt")

# OpenCV Haar face cascade (fast CPU face blur)
FACE_CASCADE = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)

# --------------------
# HTML templates
# --------------------
INDEX_HTML = """
<!doctype html><html><head><title>Campus Protest Detection</title>
<style>
 body{font-family:system-ui;margin:40px;max-width:1000px}
 .box{border:1px solid #ddd;padding:24px;border-radius:12px;margin-bottom:16px}
 label{display:inline-block;min-width:190px}
 input[type=number]{width:80px}
</style></head><body>
<h1>Campus Protest Detection (CPU prototype)</h1>
<div class="box">
  <p>Upload a short video (&lt; {{max_mb}} MB). We'll annotate detections, blur faces, create trends, and trigger alerts.</p>
  <form action="/process" method="post" enctype="multipart/form-data">
    <p><input type="file" name="video" accept="video/*" required></p>

    <p><label>YOLO confidence (0.1–0.7)</label>
       <input type="number" name="conf" step="0.05" min="0.1" max="0.7" value="0.35"></p>

    <p><label>People threshold (N)</label>
       <input type="number" name="people_thresh" min="1" max="200" value="15">
       <small>Trigger when crowd ≥ N people</small></p>

    <p><label>Min duration (seconds)</label>
       <input type="number" name="min_dur" min="1" max="600" value="5">
       <small>Trigger only if sustained</small></p>

    <p><label>Face blurring</label>
       <input type="checkbox" name="faceblur" checked> <small>Blur detected faces</small></p>

    <p><button type="submit">Process</button></p>
  </form>
</div>

<div class="box">
  <p><b>Health checks:</b>
    <a href="/test-alert" target="_blank">/test-alert</a>
  </p>
</div>
</body></html>
"""

RESULT_HTML = """
<!doctype html><html><head><title>Result</title>
<style>
 body{font-family:system-ui;margin:40px;max-width:1100px}
 .box{border:1px solid #ddd;padding:24px;border-radius:12px;margin-bottom:16px}
 table{border-collapse:collapse}
 th,td{border:1px solid #ddd;padding:8px}
 a{word-break:break-all}
</style></head><body>
<h1>Finished ✅</h1>

<div class="box">
  <p><b>Input:</b> {{in_name}}</p>
  <p><b>Output video:</b> <a href="{{out_url}}">{{out_name}}</a></p>
  <p><b>Trend CSV:</b> <a href="{{csv_url}}">{{csv_name}}</a></p>
  <p><b>Trend chart (PNG):</b> <a href="{{png_url}}">{{png_name}}</a></p>
  <p><b>Alerts JSON:</b> <a href="{{alerts_url}}">{{alerts_name}}</a></p>

  {% if s3_vid %}<p><b>S3 video link:</b> <a href="{{s3_vid}}">Download (expires)</a></p>{% endif %}
  {% if s3_csv %}<p><b>S3 CSV link:</b> <a href="{{s3_csv}}">Download (expires)</a></p>{% endif %}
  {% if s3_png %}<p><b>S3 PNG link:</b> <a href="{{s3_png}}">Download (expires)</a></p>{% endif %}
  {% if s3_json %}<p><b>S3 JSON link:</b> <a href="{{s3_json}}">Download (expires)</a></p>{% endif %}
</div>

<div class="box">
  <p><b>Frames:</b> {{frames}} &nbsp; <b>Processed:</b> {{proc_frames}} &nbsp; <b>FPS:</b> {{fps}} &nbsp; <b>Duration:</b> {{duration_s}}s</p>
  <p><b>Max people in a frame:</b> {{max_people}}</p>
  <p><b>Threshold:</b> ≥{{people_thresh}} people for ≥{{min_dur}}s</p>

  {% if alerts and alerts|length>0 %}
    <h3>Triggered alerts</h3>
    <table>
      <tr><th>#</th><th>Start (s)</th><th>End (s)</th><th>Peak people</th></tr>
      {% for a in alerts %}
        <tr><td>{{loop.index}}</td><td>{{a.start_sec}}</td><td>{{a.end_sec}}</td><td>{{a.peak}}</td></tr>
      {% endfor %}
    </table>
  {% else %}
    <p><i>No alerts triggered.</i></p>
  {% endif %}
</div>

<div class="box">
  <h3>People per second</h3>
  <img src="{{png_url}}" alt="crowd trend" style="max-width:100%;border:1px solid #eee;border-radius:8px"/>
  <p><small>CSV columns: second, people_avg, people_max</small></p>
</div>

<p><a href="/">Process another</a></p>
</body></html>
"""

# --------------------
# Small helpers
# --------------------
def allowed_file(name: str) -> bool:
    _, ext = os.path.splitext(name.lower())
    return ext in ALLOWED_EXT

def blur_faces(frame):
    """Blur faces for privacy using Haar cascade (CPU)."""
    if FACE_CASCADE.empty():
        return frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30,30))
    out = frame
    for (x,y,w,h) in faces:
        roi = out[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (35,35), 0)   # strong blur
        out[y:y+h, x:x+w] = roi
    return out

def compute_trend_and_alerts(per_frame_counts: List[int], fps: float,
                             people_thresh: int, min_dur: int,
                             csv_path: str, png_path: str) -> List[Dict]:
    """Write CSV & PNG of crowd trend; return alert windows where sec_max>=threshold for >=min_dur seconds."""
    counts = np.array(per_frame_counts, dtype=np.int32)
    ts = np.arange(len(counts)) / (fps or 25.0)
    secs = np.floor(ts).astype(int)
    max_sec = int(secs.max()) if len(secs) else 0

    sec_avg, sec_max = [], []
    for s in range(max_sec + 1):
        vals = counts[secs == s]
        sec_avg.append(float(vals.mean()) if vals.size else 0.0)
        sec_max.append(int(vals.max()) if vals.size else 0)

    # CSV
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["second", "people_avg", "people_max"])
        for s in range(max_sec + 1):
            wcsv.writerow([s, round(sec_avg[s],3), sec_max[s]])

    # PNG
    plt.figure(figsize=(10,4))
    plt.plot(range(max_sec+1), sec_avg, label="people_avg/sec")
    plt.plot(range(max_sec+1), sec_max, label="people_max/sec")
    plt.axhline(people_thresh, linestyle="--", label=f"threshold={people_thresh}")
    plt.xlabel("second"); plt.ylabel("people"); plt.title("Crowd size over time")
    plt.legend(); plt.tight_layout(); plt.savefig(png_path, dpi=120); plt.close()

    # Alerts
    alerts = []
    start = None
    for s in range(max_sec + 1):
        if sec_max[s] >= people_thresh:
            start = s if start is None else start
        else:
            if start is not None and (s - start) >= min_dur:
                peak = int(max(sec_max[start:s]))
                alerts.append({"start_sec": int(start), "end_sec": int(s-1), "peak": peak})
            start = None
    if start is not None and ((max_sec + 1) - start) >= min_dur:
        peak = int(max(sec_max[start:max_sec+1]))
        alerts.append({"start_sec": int(start), "end_sec": int(max_sec), "peak": peak})
    return alerts

def run_pipeline(in_path: str, out_path: str, csv_path: str, png_path: str, alerts_path: str,
                 conf: float = 0.35, people_thresh: int = 15, min_dur: int = 5,
                 face_blur: bool = True) -> Tuple[Dict, List[Dict]]:
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    proc_frames = 0
    max_people = 0
    per_frame_counts: List[int] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if face_blur:
            frame = blur_faces(frame)

        res = model.predict(frame, conf=conf, verbose=False)[0]
        n_people = sum(int(b.cls) == 0 for b in res.boxes)
        max_people = max(max_people, n_people)
        per_frame_counts.append(n_people)

        annotated = res.plot()
        writer.write(annotated)
        proc_frames += 1

    cap.release()
    writer.release()

    alerts = compute_trend_and_alerts(per_frame_counts, fps, people_thresh, min_dur, csv_path, png_path)

    with open(alerts_path, "w") as f:
        json.dump({
            "people_threshold": people_thresh,
            "min_duration_seconds": min_dur,
            "alerts": alerts
        }, f, indent=2)

    duration_s = round(proc_frames / (fps or 25.0), 2)
    stats = {
        "frames": frames_total or proc_frames,
        "proc_frames": proc_frames,
        "fps": round(fps, 2),
        "duration_s": duration_s,
        "max_people": int(max_people),
    }
    return stats, alerts

def send_alert_emails(alerts: List[Dict], people_thresh: int, min_dur: int,
                      out_name: str, csv_name: str, png_name: str,
                      out_url: str = "", csv_url: str = "", png_url: str = "", json_url: str = ""):
    """Send one SNS email (with S3 links if available)."""
    if not alerts:
        return
    # Combine all windows into one concise email
    windows = ", ".join([f"{a['start_sec']}-{a['end_sec']}s" for a in alerts])
    peak_all = max([a["peak"] for a in alerts]) if alerts else 0
    subject = f"Crowd alert: ≥{people_thresh} for ≥{min_dur}s ({len(alerts)} window{'s' if len(alerts)!=1 else ''})"
    lines = [
        "[Campus AI] Crowd alert triggered",
        f"Threshold: ≥{people_thresh} people for ≥{min_dur}s",
        f"Windows: {windows}",
        f"Peak people: {peak_all}",
        f"Video: {out_name}",
        f"CSV: {csv_name}",
        f"Chart: {png_name}",
    ]
    if out_url:  lines.append(f"Video link: {out_url}")
    if csv_url:  lines.append(f"CSV link:   {csv_url}")
    if png_url:  lines.append(f"Chart link: {png_url}")
    if json_url: lines.append(f"JSON link:  {json_url}")
    ok, info = publish_sns(subject, "\n".join(lines))
    print("SNS:", ok, info)

# --------------------
# Routes
# --------------------
@app.get("/")
def index():
    return render_template_string(INDEX_HTML, max_mb=MAX_MB)

@app.get("/outputs/<path:fname>")
def download(fname):
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(OUTPUT_DIR, fname, as_attachment=False)

@app.get("/test-alert")
def test_alert():
    ok, info = publish_sns("Crowd alert: TEST",
                           "[Campus AI] TEST alert\nThreshold: ≥6 for ≥5s\nWindow: 10-20s\nPeak: 12\n")
    return (f"SNS publish: {ok}, {info}\n", 200) if ok else (f"SNS publish failed: {info}\n", 500)

@app.post("/process")
def process():
    if "video" not in request.files:
        abort(400, "No file part")
    f = request.files["video"]
    if f.filename == "":
        abort(400, "No selected file")
    if not allowed_file(f.filename):
        abort(400, f"Unsupported extension. Allowed: {', '.join(sorted(ALLOWED_EXT))}")

    conf = max(0.1, min(0.7, request.form.get("conf", type=float, default=0.35)))
    people_thresh = request.form.get("people_thresh", type=int, default=15)
    min_dur = request.form.get("min_dur", type=int, default=5)
    use_faceblur = bool(request.form.get("faceblur"))

    uid = uuid.uuid4().hex[:12]
    in_name = f"{uid}_{os.path.basename(f.filename)}"
    in_path = os.path.join(UPLOAD_DIR, in_name)
    f.save(in_path)

    base = in_name.rsplit(".", 1)[0]
    out_name = base + "_annotated.mp4"
    csv_name = base + "_trend.csv"
    png_name = base + "_trend.png"
    alerts_name = base + "_alerts.json"

    out_path = os.path.join(OUTPUT_DIR, out_name)
    csv_path = os.path.join(OUTPUT_DIR, csv_name)
    png_path = os.path.join(OUTPUT_DIR, png_name)
    alerts_path = os.path.join(OUTPUT_DIR, alerts_name)

    stats, alerts = run_pipeline(
        in_path, out_path, csv_path, png_path, alerts_path,
        conf=conf, people_thresh=people_thresh, min_dur=min_dur,
        face_blur=use_faceblur
    )

    # Upload artifacts to S3 + pre-sign
    run_id  = uuid.uuid4().hex[:12]
    s3_vid  = upload_and_sign(out_path,    run_id)
    s3_csv  = upload_and_sign(csv_path,    run_id)
    s3_png  = upload_and_sign(png_path,    run_id)
    s3_json = upload_and_sign(alerts_path, run_id)

    # Send one consolidated alert email with links
    send_alert_emails(
        alerts=alerts, people_thresh=people_thresh, min_dur=min_dur,
        out_name=os.path.basename(out_path), csv_name=os.path.basename(csv_path),
        png_name=os.path.basename(png_path),
        out_url=s3_vid, csv_url=s3_csv, png_url=s3_png, json_url=s3_json
    )

    return render_template_string(
        RESULT_HTML,
        in_name=os.path.basename(in_path),
        out_name=os.path.basename(out_path), out_url=f"/outputs/{os.path.basename(out_path)}",
        csv_name=os.path.basename(csv_path), csv_url=f"/outputs/{os.path.basename(csv_path)}",
        png_name=os.path.basename(png_path), png_url=f"/outputs/{os.path.basename(png_path)}",
        alerts_name=os.path.basename(alerts_path), alerts_url=f"/outputs/{os.path.basename(alerts_path)}",
        frames=stats["frames"], proc_frames=stats["proc_frames"], fps=stats["fps"],
        duration_s=stats["duration_s"], max_people=stats["max_people"],
        people_thresh=people_thresh, min_dur=min_dur, alerts=alerts,
        s3_vid=s3_vid, s3_csv=s3_csv, s3_png=s3_png, s3_json=s3_json
    )

# --------------------
# Main
# --------------------
if __name__ == "__main__":
    # Run development server (behind Security Group rule opening TCP/5000 to your IP)
    app.run(host="0.0.0.0", port=5000)
