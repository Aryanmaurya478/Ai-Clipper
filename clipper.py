"""
AI Clipper engine: turn a long video link into one Instagram Reel (9:16, captions burned in).

Steps:
  1. Download the video (yt-dlp)
  2. Transcribe speech with word timestamps (faster-whisper, runs locally, free)
  3. Pick the best 15-60s moment (Claude if an API key is set, otherwise a free built-in picker)
  4. Cut it, crop to 1080x1920 vertical, burn in word-by-word captions (ffmpeg)

Used by app.py (web interface). Can also run from the command line:
  python clipper.py "https://www.youtube.com/watch?v=..."
  python clipper.py "<link>" --start 120 --end 165
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def _ffmpeg_dir() -> str:
    """yt-dlp only recognises a file literally named ffmpeg.exe, but imageio's file is called
    ffmpeg-win-x86_64-v7.1.exe. Make bin/ffmpeg.exe (a hard link = no extra disk space)."""
    bin_dir = Path(__file__).parent / "bin"
    target = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not target.exists() or target.stat().st_size != os.path.getsize(FFMPEG):
        bin_dir.mkdir(exist_ok=True)
        target.unlink(missing_ok=True)
        try:
            os.link(FFMPEG, target)
        except OSError:
            shutil.copy2(FFMPEG, target)  # different drive: copy instead
    # yt-dlp's partial-download check ignores ffmpeg_location and only searches PATH
    if str(bin_dir) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    return str(bin_dir)


def _enable_nvidia_dlls():
    """Let Whisper find the CUDA DLLs installed by pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    try:
        import nvidia
    except ImportError:
        return
    for base in list(nvidia.__path__):
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            d = os.path.join(base, sub, "bin")
            if os.path.isdir(d):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(d)


_enable_nvidia_dlls()
MODEL = "claude-opus-5-5"


def _print(msg):
    if msg.startswith("PROGRESS "):  # one updating line in the terminal instead of hundreds
        print("\r  " + msg[9:] + "   ", end="", flush=True)
    else:
        print(msg, flush=True)


# ---------------------------------------------------------------- 1. download
def _ydl_opts(work: Path, name: str, fmt: str, log, label: str) -> dict:
    def hook(d):
        if d.get("status") == "downloading":
            pct = re.sub(r"\x1b\[[0-9;]*m", "", d.get("_percent_str", "")).strip()  # drop colour codes
            log(f"PROGRESS {label} {pct}")

    return {
        "format": fmt,
        "outtmpl": str(work / f"{name}.%(ext)s"),
        "ffmpeg_location": _ffmpeg_dir(),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,  # our hook reports progress; stop yt-dlp printing its own bar
        "no_warnings": True,
        "progress_hooks": [hook],
    }


def _find(work: Path, name: str) -> Path:
    files = [p for p in work.glob(f"{name}.*") if not p.name.endswith((".part", ".ytdl"))]
    if not files:
        raise RuntimeError(f"Download failed ({name})")
    return max(files, key=lambda p: p.stat().st_size)


def download_audio(url: str, work: Path, log=_print) -> Path:
    """Download only the sound track (a few MB) - enough to find the best moment."""
    import yt_dlp

    opts = _ydl_opts(work, "audio", "ba[ext=m4a]/ba/b[height<=360]/b", log, "audio")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    log(f"Got audio for: {info.get('title')}")
    return _find(work, "audio")


QUALITY = {  # name -> (max source height to download, output width, output height)
    "best": (2160, 1080, 1920),  # 4K source = sharpest vertical crop
    "1080": (1080, 1080, 1920),
    "720": (720, 720, 1280),
    "480": (480, 480, 854),
}


def download_section(url: str, work: Path, start: float, end: float, max_h: int = 1080, log=_print) -> Path:
    """Download only the chosen seconds of the video, not the whole thing."""
    import yt_dlp
    from yt_dlp.utils import download_range_func

    fmt = f"bv*[height<={max_h}]+ba/b[height<={max_h}]/b"
    opts = _ydl_opts(work, "section", fmt, log, "clip")
    opts.update({
        "merge_output_format": "mp4",
        "download_ranges": download_range_func(None, [(start, end)]),
        "force_keyframes_at_cuts": True,  # exact cut points
    })
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    log(f"Downloaded only {end - start:.0f}s of video at {info.get('height') or '?'}p")
    return _find(work, "section")


# ---------------------------------------------------------------- 2. transcribe
def sentences_from_words(words, max_len=10.0, gap=0.7):
    """Group words into short sentences: break at a pause, a full stop (. ? ! ।) or after ~10s.
    Gives the AI many exact cut points instead of Whisper's long 30s chunks."""
    segs, cur = [], []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > gap or w["end"] - cur[0]["start"] > max_len):
            segs.append(cur); cur = []
        cur.append(w)
        if w["word"].endswith((".", "?", "!", "।")):
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)
    return [{"start": round(g[0]["start"], 2), "end": round(g[-1]["end"], 2),
             "text": " ".join(x["word"] for x in g)} for g in segs]


def transcribe(video: Path, work: Path, model_size: str, log=_print):
    from faster_whisper import WhisperModel

    import numpy as np

    # Decode audio with ffmpeg into raw 16 kHz mono floats. We hand Whisper a numpy array
    # instead of a file path, so it never uses PyAV (newer PyAV versions break faster-whisper).
    raw = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout
    audio = np.frombuffer(raw, dtype=np.float32)
    log(f"Audio length: {len(audio) / 16000:.0f}s")

    log(f"Transcribing with whisper '{model_size}' (first run downloads the model)...")

    def run(device, compute_type):
        from faster_whisper import BatchedInferencePipeline

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        if device == "cuda":
            # Batched mode: the GPU works on 8 chunks of audio at once - several times faster.
            segments, info = BatchedInferencePipeline(model).transcribe(
                audio, batch_size=8, word_timestamps=True, vad_filter=True)
        else:
            segments, info = model.transcribe(
                audio, word_timestamps=True, vad_filter=True,
                condition_on_previous_text=False)  # stops slow repeat loops on noisy gaming audio
        words = []
        for s in segments:  # transcription actually happens here, while looping
            for w in s.words or []:
                if w.word.strip():
                    words.append({"start": float(w.start), "end": float(w.end), "word": w.word.strip()})
            log(f"  [{s.start:7.1f}s] {s.text.strip()}")
        segs = sentences_from_words(words)
        log(f"Language: {info.language}, {len(segs)} sentences, {len(words)} words")
        return segs, words

    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            log("Using NVIDIA GPU (fast)")
            return run("cuda", "float16")
        return run("cpu", "int8")
    except Exception as ex:
        msg = str(ex).lower()
        if not any(k in msg for k in ("cublas", "cudnn", "cuda", ".dll", "gpu")):
            raise
        log("GPU not ready (CUDA files missing), so using the CPU instead. This is slower but works.")
        return run("cpu", "int8")


# ---------------------------------------------------------------- 3. pick the moment
CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "start": {"type": "number", "description": "clip start in seconds"},
        "end": {"type": "number", "description": "clip end in seconds"},
        "hook": {"type": "string", "description": "short on-screen hook title, max 6 words"},
        "reason": {"type": "string", "description": "why this moment will perform well"},
        "caption": {"type": "string", "description": "Instagram post caption, 1-3 lines"},
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["start", "end", "hook", "reason", "caption", "hashtags"],
    "additionalProperties": False,
}


SETTINGS_FILE = Path(__file__).parent / "settings.json"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(new: dict):
    s = load_settings()
    s.update({k: v for k, v in new.items() if v is not None})
    SETTINGS_FILE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def gemini_key() -> str:
    return load_settings().get("gemini_key") or os.environ.get("GEMINI_API_KEY", "")


def has_claude_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def provider() -> str:
    """Which AI picks the clip: Gemini if its key is saved, else Claude if its key is set, else free."""
    if gemini_key():
        return "gemini"
    if has_claude_key():
        return "claude"
    return "free"


def build_prompt(segs, min_len: int, max_len: int) -> str:
    transcript = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segs)
    return (
        "You are an expert short-form video editor for Instagram Reels.\n"
        f"From the timestamped transcript below, choose the ONE best clip between {min_len} and {max_len} seconds long.\n"
        "Pick a moment with a strong hook in the first 3 seconds, a complete thought (it must make sense with no context), "
        "and an emotional, funny, surprising or highly useful payoff. Start and end on sentence boundaries "
        "using the segment timestamps. Write the hook and caption in the same language as the video.\n\n"
        f"<transcript>\n{transcript}\n</transcript>"
    )


def _model_rank(name: str):
    """Sort best first: highest version, then pro > flash > flash-lite, stable before preview."""
    n = name.lower()
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
    version = float(m.group(1)) if m else 0
    tier = 0 if "pro" in n else 2 if "lite" in n else 1 if "flash" in n else 3
    unstable = any(x in n for x in ("preview", "exp"))
    return (-version, tier, unstable, n)


def list_gemini_models() -> list:
    """Ask Google which Gemini text models this key can use."""
    from google import genai

    client = genai.Client(api_key=gemini_key())
    out = []
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        actions = getattr(m, "supported_actions", None) or []
        if "gemini" not in name or (actions and "generateContent" not in actions):
            continue
        if any(x in name for x in ("embedding", "image", "tts", "audio", "live", "veo", "imagen")):
            continue
        out.append({"id": name, "label": getattr(m, "display_name", None) or name})
    out.sort(key=lambda x: _model_rank(x["id"]))
    return out


def pick_with_gemini(segs, min_len: int, max_len: int, log=_print) -> dict:
    """Ask Gemini. If Google is busy (503/429), retry, then try other models."""
    import time
    from google import genai

    chosen = load_settings().get("gemini_model") or DEFAULT_GEMINI_MODEL
    prompt = build_prompt(segs, min_len, max_len) + (
        "\n\nReply with ONLY a JSON object with these keys: "
        '"start" (number, seconds), "end" (number, seconds), "hook" (max 6 words), '
        '"reason" (string), "caption" (Instagram caption, 1-3 lines), "hashtags" (array of strings, no #).'
    )
    client = genai.Client(api_key=gemini_key())

    models = [chosen]
    try:  # backups: other good text models this key can use
        models += [m["id"] for m in list_gemini_models() if m["id"] != chosen][:3]
    except Exception:
        pass

    last_err = None
    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model, contents=prompt, config={"response_mime_type": "application/json"},
                )
                if model != chosen:
                    log(f"Used backup model {model}")
                return _parse_gemini(response.text)
            except Exception as ex:
                last_err = ex
                busy = any(c in str(ex) for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "overloaded"))
                if not busy:
                    raise
                wait = 4 * (attempt + 1)
                log(f"Gemini {model} is busy, retrying in {wait}s...")
                time.sleep(wait)
    raise RuntimeError(f"Gemini is busy right now: {last_err}")


def _parse_gemini(text) -> dict:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text).strip()
    try:
        clip = json.loads(text)
    except ValueError:
        raise RuntimeError(f"Gemini gave an answer that is not JSON: {text[:200]}")
    if isinstance(clip, list):
        clip = clip[0]
    clip["start"], clip["end"] = float(clip["start"]), float(clip["end"])
    clip.setdefault("hook", ""); clip.setdefault("reason", ""); clip.setdefault("caption", "")
    clip.setdefault("hashtags", [])
    return clip


def pick_with_claude(segs, min_len: int, max_len: int) -> dict:
    import anthropic

    prompt = build_prompt(segs, min_len, max_len)
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": CLIP_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("Claude declined this video. Set the start/end time manually.")
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


HOOK_WORDS = {"secret", "never", "always", "why", "how", "mistake", "truth", "best", "worst",
              "money", "crazy", "insane", "important", "nobody", "everyone", "stop", "free"}


def pick_free(segs, min_len: int, max_len: int) -> dict:
    """No API key: score every window of whole segments by energy (words/sec, hook words, ? and !)."""
    best, best_score = None, -1.0
    for i in range(len(segs)):
        for j in range(i, len(segs)):
            dur = segs[j]["end"] - segs[i]["start"]
            if dur > max_len:
                break
            if dur < min_len:
                continue
            text = " ".join(s["text"] for s in segs[i:j + 1])
            words = text.split()
            score = len(words) / dur
            score += 0.6 * sum(w.strip(".,!?").lower() in HOOK_WORDS for w in words)
            score += 0.8 * text.count("?") + 0.5 * text.count("!")
            first = segs[i]["text"]
            score += 1.5 if first.endswith("?") or any(w.lower().strip(".,!?") in HOOK_WORDS for w in first.split()[:6]) else 0
            if score > best_score:
                best, best_score = (i, j, text), score
    if best is None:  # video shorter than min_len
        best = (0, len(segs) - 1, " ".join(s["text"] for s in segs))
    i, j, text = best
    hook = " ".join(segs[i]["text"].split()[:6])
    return {"start": segs[i]["start"], "end": segs[j]["end"], "hook": hook,
            "reason": "Free picker: most energetic, hook-heavy stretch of speech.",
            "caption": text[:150] + ("..." if len(text) > 150 else ""), "hashtags": ["reels", "viral", "fyp"]}


def snap_to_words(start, end, words):
    """Nudge start/end so we never cut in the middle of a word."""
    if not words:
        return start, end
    s = min(words, key=lambda w: abs(w["start"] - start))["start"]
    e = min(words, key=lambda w: abs(w["end"] - end))["end"]
    return max(0, s - 0.15), e + 0.3


# ---------------------------------------------------------------- 4. render
def ass_time(t):
    h, rem = divmod(max(t, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def make_subtitles(words, start, end, hook, path: Path, words_per_line=3):
    clip_words = [w for w in words if w["end"] > start and w["start"] < end]
    all_text = hook + " ".join(w["word"] for w in clip_words)
    # Arial Black has no Hindi/Devanagari or other Indic letters; Nirmala UI (built into Windows) does.
    font = "Nirmala UI" if re.search(r"[ऀ-෿]", all_text) else "Arial Black"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,78,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1
Style: Hook,Arial Black,70,&H00000000,&H0000FFFF,&H0000FFFF,&H0000FFFF,1,0,0,0,100,100,0,0,3,18,0,8,60,60,180,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".replace("Arial Black", font)
    clean = lambda s: re.sub(r"[{}\\]", "", s).upper()
    lines = [f"Dialogue: 1,{ass_time(0)},{ass_time(3.5)},Hook,,0,0,0,,{clean(hook)}"] if hook else []
    for i in range(0, len(clip_words), words_per_line):
        group = clip_words[i:i + words_per_line]
        g_end = group[-1]["end"] - start
        for j, w in enumerate(group):  # highlight the word being spoken in yellow
            w_start = w["start"] - start
            w_end = group[j + 1]["start"] - start if j + 1 < len(group) else g_end
            parts = ["{\\c&H00FFFF&}" + clean(x["word"]) + "{\\c&HFFFFFF&}" if k == j else clean(x["word"])
                     for k, x in enumerate(group)]
            lines.append(f"Dialogue: 0,{ass_time(w_start)},{ass_time(w_end)},Cap,,0,0,0,,{' '.join(parts)}")
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def render(video: Path, start, end, subs: Path, out: Path, captions=True, size=(1080, 1920)):
    vf = f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale={size[0]}:{size[1]}:flags=lanczos,setsar=1"
    if captions:
        vf += f",subtitles={subs.name}"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "error",
        "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", str(video.resolve()),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-r", "30", "-movflags", "+faststart",
        str(out.resolve()),
    ], check=True, cwd=subs.parent)  # cwd trick avoids Windows path escaping in the subtitles filter


# ---------------------------------------------------------------- full pipeline
def make_reel(url, out_dir: Path, start=None, end=None, min_len=20, max_len=60,
              whisper="small", captions=True, quality="1080", log=_print) -> dict:
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    try:
        return _make_reel(url, out_dir, work, start, end, min_len, max_len, whisper, captions,
                          QUALITY.get(str(quality), QUALITY["1080"]), log)
    finally:
        # Delete everything that was downloaded / temporary. Only reel.mp4 + caption.txt stay.
        # (A local file you passed in is never touched - it is not inside work/.)
        shutil.rmtree(work, ignore_errors=True)
        log("Cleaned up: downloaded audio and video deleted")


def _make_reel(url, out_dir, work, start, end, min_len, max_len, whisper, captions, quality, log):
    max_h, out_w, out_h = quality
    local = os.path.exists(url)

    log("STEP 1/4 Downloading audio only")
    source = Path(url) if local else download_audio(url, work, log)

    import time
    t0 = time.time()
    log("STEP 2/4 Transcribing speech")
    segs, words = transcribe(source, work, whisper, log)
    log(f"Transcribing took {time.time() - t0:.0f}s")

    t0 = time.time()
    log("STEP 3/4 Finding the best moment")
    if start is not None and end is not None:
        clip = {"start": start, "end": end, "hook": "", "reason": "Chosen manually",
                "caption": "", "hashtags": []}
    elif not segs:
        raise RuntimeError("No speech found in this video. Enter a start and end time manually.")
    elif provider() in ("gemini", "claude"):
        name = provider().title()
        log(f"Asking {name} AI to pick the most viral moment...")
        try:
            clip = (pick_with_gemini(segs, min_len, max_len, log) if name == "Gemini"
                    else pick_with_claude(segs, min_len, max_len))
        except Exception as ex:  # AI down / key problem: still make a Reel
            log(f"{name} failed ({str(ex)[:150]}). Using the free built-in picker instead.")
            clip = pick_free(segs, min_len, max_len)
    else:
        log("No AI key saved, so using the free built-in picker")
        clip = pick_free(segs, min_len, max_len)
    # Safety check: AI answers can be off - keep the clip inside the video and within length limits.
    total = segs[-1]["end"] if segs else clip["end"]
    clip["start"] = max(0.0, min(float(clip["start"]), total - 1))
    clip["end"] = min(float(clip["end"]), total + 1)
    if start is None:  # only for AI/free picks, not manual times
        if clip["end"] - clip["start"] < min_len:
            clip["end"] = min(clip["start"] + min_len, total + 1)
        if clip["end"] - clip["start"] > max_len:
            clip["end"] = clip["start"] + max_len
    if clip["end"] <= clip["start"]:
        raise RuntimeError(f"Bad clip time {clip['start']}s -> {clip['end']}s")
    log(f"Picked {clip['start']:.1f}s -> {clip['end']:.1f}s : {clip['hook']}  (took {time.time() - t0:.0f}s)")

    log("STEP 4/4 Downloading just the clip and rendering 9:16 Reel")
    s, e = snap_to_words(clip["start"], clip["end"], words)
    subs = work / "captions.ass"
    make_subtitles(words, s, e, clip["hook"], subs)
    out_file = out_dir / "reel.mp4"
    log(f"Output size: {out_w}x{out_h}")
    if local:
        render(source, s, e, subs, out_file, captions, (out_w, out_h))
    else:
        section = download_section(url, work, s, e, max_h, log)
        render(section, 0, e - s, subs, out_file, captions, (out_w, out_h))  # section already starts at s

    tags = " ".join("#" + h.lstrip("#") for h in clip["hashtags"])
    (out_dir / "caption.txt").write_text(f"{clip['caption']}\n\n{tags}\n", encoding="utf-8")
    log(f"Done! Saved {out_file.resolve()}")
    return {**clip, "start": s, "end": e, "file": str(out_file.resolve())}


def main():
    ap = argparse.ArgumentParser(description="Make an Instagram Reel from a video link")
    ap.add_argument("url", help="video link (YouTube, etc.) or a local file path")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--whisper", default="small", help="tiny/base/small/medium/large-v3")
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--quality", default="1080", choices=list(QUALITY))
    ap.add_argument("--out", default="output")
    a = ap.parse_args()
    try:
        make_reel(a.url, Path(a.out), a.start, a.end, a.min, a.max, a.whisper, not a.no_captions, a.quality)
    except Exception as ex:
        sys.exit("Error: " + re.sub(r"\x1b\[[0-9;]*m", "", str(ex)))


if __name__ == "__main__":
    main()
