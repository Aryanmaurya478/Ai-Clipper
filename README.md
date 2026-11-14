# AI Reel Clipper

Paste a video link and get one Instagram Reel (1080x1920, 9:16) with word-by-word captions, plus a ready-to-post caption and hashtags.

## How it works
1. **Download**: yt-dlp fetches the video (YouTube and most other sites)
2. **Transcribe**: faster-whisper runs on your PC for free
3. **Pick the best moment**: Claude AI if `ANTHROPIC_API_KEY` is set, otherwise a free built-in picker
4. **Render**: ffmpeg crops to vertical and burns in the captions

## Run it in Antigravity (or VS Code)
1. Open this folder: **File > Open Folder > Video editer**
2. Install once: **Terminal > Run Task > Install requirements**
3. Press **F5** ("Run AI Reel Clipper"), or **Ctrl+Shift+B**
4. Your browser opens at http://localhost:5000. Paste a link and click **Make my Reel**

You can also ask the Antigravity agent: *"run app.py"*.

## Run from a terminal
```
pip install -r requirements.txt
python app.py
```

## Optional: use Claude AI to choose the clip
Get a key at https://console.anthropic.com, then set it before starting:
```
setx ANTHROPIC_API_KEY "sk-ant-..."
```
Restart Antigravity or the terminal after running `setx`.

Output files are saved in `output/<job-id>/reel.mp4` and `caption.txt`.
Only clip videos you own or have permission to reuse.
