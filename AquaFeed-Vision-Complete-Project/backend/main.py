import json
import uuid
import re
import html
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import (
    FastAPI, HTTPException, UploadFile, File,
    Form, BackgroundTasks
)
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import BASE, RUNS, RUNTIME, system_info
from .analysis import analyse_video

app = FastAPI(title="AquaFeed Vision", version="2.0")

app.mount(
    "/assets",
    StaticFiles(directory=str(BASE / "frontend")),
    name="assets"
)

JOBS = {}
ANALYSIS_LOCK = threading.Lock()
FEED_LOCK = threading.Lock()
FEED_FILE = RUNTIME / "feeding_history.json"


def now():
    return datetime.now(timezone.utc)


def read_feed():
    if not FEED_FILE.exists():
        return {"last_fed_at": None, "interval_hours": 12, "events": []}
    try:
        value = json.loads(FEED_FILE.read_text("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Bad record")
        return value
    except (json.JSONDecodeError, ValueError):
        return {"last_fed_at": None, "interval_hours": 12, "events": []}


def save_feed(value):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    temp = FEED_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(FEED_FILE)


def feed_state():
    record = read_feed()
    last = record.get("last_fed_at")
    interval = float(record.get("interval_hours", 12))
    if not last:
        return {
            "state": "NO_FEEDING_HISTORY",
            "label": "No feeding record",
            "last_fed_at": None,
            "hours_since_feed": None,
            "next_check_at": None,
            "interval_hours": interval,
            "is_due": False
        }

    dt = datetime.fromisoformat(last.replace("Z", "+00:00"))
    elapsed = (now() - dt).total_seconds() / 3600
    next_check = dt + timedelta(hours=interval)
    due = elapsed >= interval

    return {
        "state": "SCHEDULE_DUE" if due else "RECENTLY_FED",
        "label": "Feeding check due" if due else "Recently fed",
        "last_fed_at": last,
        "hours_since_feed": round(max(0, elapsed), 2),
        "next_check_at": next_check.isoformat(),
        "interval_hours": interval,
        "is_due": due
    }


class FeedSettings(BaseModel):
    interval_hours: float = Field(ge=1, le=48)


@app.get("/")
def home():
    return FileResponse(BASE / "frontend" / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "service": "AquaFeed Vision"}


@app.get("/api/system")
def system():
    return system_info()


@app.get("/api/feeding")
def feeding():
    return feed_state()


@app.post("/api/feeding/settings")
def update_settings(settings: FeedSettings):
    with FEED_LOCK:
        item = read_feed()
        item["interval_hours"] = settings.interval_hours
        save_feed(item)
        return feed_state()


@app.post("/api/feeding/mark-fed")
def mark_fed():
    """Log a real manually performed feeding; no hardware command."""
    with FEED_LOCK:
        item = read_feed()
        timestamp = now().isoformat()
        item["last_fed_at"] = timestamp
        item.setdefault("events", []).append({
            "timestamp": timestamp,
            "source": "user_confirmed"
        })
        save_feed(item)
        return feed_state()


def valid_id(value):
    if not re.fullmatch(r"[a-f0-9-]{36}", value):
        raise HTTPException(400, "Invalid analysis ID")


def work(job_id, video, confidence, sample_count):
    directory = RUNS / job_id

    def progress(message, percentage):
        JOBS[job_id]["message"] = message
        JOBS[job_id]["progress"] = percentage

    with ANALYSIS_LOCK:
        JOBS[job_id]["state"] = "processing"
        try:
            result = analyse_video(
                video, directory,
                confidence, sample_count, progress
            )
            result["job_id"] = job_id
            (directory / "result.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            JOBS[job_id].update({
                "state": "done",
                "progress": 100,
                "message": "Analysis completed",
                "result": result
            })
        except Exception as exc:
            JOBS[job_id].update({
                "state": "failed",
                "progress": 0,
                "message": str(exc)[:500]
            })


@app.post("/api/analyse", status_code=202)
async def analyse(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    confidence: float = Form(0.25),
    sample_count: int = Form(12)
):
    if not 0.10 <= confidence <= 0.90:
        raise HTTPException(422, "Confidence out of range")
    if sample_count not in (12, 24, 36):
        raise HTTPException(422, "Invalid sample count")

    extension = Path(video.filename or "clip.mp4").suffix.lower()
    if extension not in (
        ".mp4", ".avi", ".mov", ".webm", ".mkv"
    ):
        raise HTTPException(400, "Unsupported video type")

    job_id = str(uuid.uuid4())
    directory = RUNS / job_id
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / ("video" + extension)

    size = 0
    try:
        with saved.open("wb") as f:
            while True:
                block = await video.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > 250 * 1024 * 1024:
                    raise HTTPException(413, "Max video size: 250MB")
                f.write(block)
    except Exception:
        saved.unlink(missing_ok=True)
        raise
    finally:
        await video.close()

    if not size:
        raise HTTPException(400, "Empty video file")

    JOBS[job_id] = {
        "state": "queued", "progress": 2,
        "message": "Video uploaded successfully"
    }
    background_tasks.add_task(
        work, job_id, saved, confidence, sample_count
    )
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    valid_id(job_id)
    if job_id in JOBS:
        return JOBS[job_id]

    path = RUNS / job_id / "result.json"
    if path.exists():
        return {
            "state": "done",
            "progress": 100,
            "message": "Saved analysis",
            "result": json.loads(path.read_text("utf-8"))
        }
    raise HTTPException(404, "Analysis ID not found")


@app.get("/api/detection-image/{job_id}")
def detection_image(job_id: str):
    valid_id(job_id)
    path = RUNS / job_id / "detections.jpg"
    if not path.exists():
        raise HTTPException(404, "Image not available")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/report/{job_id}")
def download_report(job_id: str):
    valid_id(job_id)
    file = RUNS / job_id / "result.json"
    if not file.exists():
        raise HTTPException(404, "Report unavailable")

    data = json.loads(file.read_text("utf-8"))
    fish = data["fish"]
    behavior = data["behaviour"]

    fields = [
        ("Visible fish (median)", fish.get("count_median")),
        ("Maximum visible fish", fish.get("count_max")),
        ("Activity index", behavior.get("movement_index")),
        ("Activity description", behavior.get("movement_level")),
        ("Behaviour difference", behavior.get("behaviour_deviation")),
        ("Autoencoder MSE", behavior.get("mse")),
        ("Video duration", data["video"].get("duration_seconds")),
        ("YOLO confidence threshold", data["settings"].get("confidence")),
        ("Hunger validation", "Not validated"),
        ("Health status", "Not medically assessed")
    ]

    rows = "".join(
        "<tr><th>" + html.escape(str(a)) + "</th><td>" +
        html.escape(str(b if b is not None else "N/A")) +
        "</td></tr>"
        for a, b in fields
    )

    page = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>AquaFeed Vision Analysis Report</title>
<style>
body{font:15px Arial,sans-serif;max-width:840px;margin:45px auto;color:#223}
header{background:#182428;color:white;padding:26px;border-radius:8px}
h1{margin:0;color:#ffb32c}h2{margin-top:28px}
table{width:100%;border-collapse:collapse;margin-top:20px}
th,td{padding:13px;text-align:left;border-bottom:1px solid #ddd}
th{background:#f8f8f5;width:50%}
.notice{background:#fff4df;padding:15px;border-radius:8px;margin-top:22px}
@media print{body{margin:10mm}}
</style></head><body>
<header><h1>AquaFeed Vision</h1>
<p>Goldfish AI — Video Analysis Report</p></header>
<h2>Analysis Summary</h2><table>""" + rows + """</table>
<div class="notice"><strong>Research limitation:</strong>
YOLO measures visible detections, not confirmed total unique fish.
Scene motion is not calibrated individual swimming speed.
No validated physiological hunger or health diagnosis is available.
Feeding reminders follow the user-recorded schedule.</div>
<p>Report generated by AquaFeed Vision.</p>
<p>Open this HTML report in a browser and select Print → Save as PDF
to keep a PDF copy.</p></body></html>"""

    report = RUNS / job_id / "AquaFeed_Analysis_Report.html"
    report.write_text(page, encoding="utf-8")
    return FileResponse(
        report, media_type="text/html",
        filename="AquaFeed_Analysis_Report.html"
    )
