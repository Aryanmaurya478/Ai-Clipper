"""
Web interface for the AI Clipper.
Run:  python app.py   then open http://localhost:5000
"""

import re
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

import clipper

ROOT = Path(__file__).parent
app = Flask(__name__, static_folder=str(ROOT / "static"))
jobs = {}  # job_id -> {"status", "log", "result", "error"}


def run_job(job_id, params):
    job = jobs[job_id]

    def log(msg):
        msg = re.sub(r"\x1b\[[0-9;]*m", "", str(msg))  # remove terminal colour codes like [0;31m
        clipper._print(msg)
        if msg.startswith("PROGRESS "):  # replace last progress line instead of spamming
            if job["log"] and job["log"][-1].startswith("PROGRESS "):
                job["log"][-1] = msg
                return
        job["log"].append(msg)

    try:
        job["result"] = clipper.make_reel(log=log, out_dir=ROOT / "output" / job_id, **params)
        job["status"] = "done"
    except Exception as ex:
        job["error"] = re.sub(r"\x1b\[[0-9;]*m", "", str(ex))
        job["status"] = "error"
        log(f"ERROR: {ex}")


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/info")
def info():
    s = clipper.load_settings()
    return jsonify({
        "provider": clipper.provider(),
        "gemini_saved": bool(clipper.gemini_key()),
        "gemini_model": s.get("gemini_model") or clipper.DEFAULT_GEMINI_MODEL,
    })


@app.get("/api/gemini_models")
def gemini_models():
    if not clipper.gemini_key():
        return jsonify({"error": "Save your Gemini key first"}), 400
    try:
        return jsonify({"models": clipper.list_gemini_models()})
    except Exception as ex:
        return jsonify({"error": f"Could not load models: {ex}"}), 400


@app.post("/api/settings")
def settings():
    d = request.get_json(force=True)
    new = {}
    if "gemini_key" in d:  # empty string = remove key
        new["gemini_key"] = (d["gemini_key"] or "").strip()
    if d.get("gemini_model"):
        new["gemini_model"] = d["gemini_model"].strip()
    clipper.save_settings(new)
    return info()


@app.post("/api/clip")
def clip():
    d = request.get_json(force=True)
    url = (d.get("url") or "").strip()
    if not url:
        return jsonify({"error": "Paste a video link first"}), 400
    num = lambda k: float(d[k]) if d.get(k) not in (None, "") else None
    params = {
        "url": url,
        "start": num("start"),
        "end": num("end"),
        "min_len": int(d.get("min_len") or 20),
        "max_len": int(d.get("max_len") or 60),
        "whisper": d.get("whisper") or "small",
        "captions": bool(d.get("captions", True)),
        "quality": str(d.get("quality") or "1080"),
    }
    job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    jobs[job_id] = {"status": "running", "log": [], "result": None, "error": None}
    threading.Thread(target=run_job, args=(job_id, params), daemon=True).start()
    return jsonify({"job": job_id})


@app.get("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.get("/api/video/<job_id>")
def video(job_id):
    path = ROOT / "output" / job_id / "reel.mp4"
    if not path.exists():
        return "not ready", 404
    return send_file(path, mimetype="video/mp4", as_attachment=request.args.get("dl") == "1",
                     download_name=f"reel-{job_id}.mp4", conditional=True)


if __name__ == "__main__":
    print("AI Clipper running at http://localhost:5000")
    threading.Timer(1.2, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(port=5000, debug=False, threaded=True)
