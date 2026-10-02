"""Verify the rendered film against the timeline.

    .venv-tools/bin/python tools/verify_render.py [out/cifar-journey.mp4]

Checks resolution, fps, duration vs src/data/timeline.json, presence and level
of narration audio, long silences, and writes out/mp4-contact-sheet.jpg with
one decoded frame from the middle of every chapter.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

VIDEO = Path(__file__).resolve().parents[1]
FF = imageio_ffmpeg.get_ffmpeg_exe()
mp4 = Path(sys.argv[1]) if len(sys.argv) > 1 else VIDEO / "out" / "cifar-journey.mp4"
tl = json.loads((VIDEO / "src" / "data" / "timeline.json").read_text())

info = subprocess.run([FF, "-i", str(mp4)], capture_output=True, text=True).stderr
dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info)
seconds = int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3])
video = re.search(r"Video: (\w+).*?, (\d+)x(\d+).*?, ([\d.]+) fps", info)
audio = re.search(r"Audio: (\w+).*?, (\d+) Hz", info)
expected = tl["durationInFrames"] / tl["fps"]

vol = subprocess.run([FF, "-i", str(mp4), "-vn", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True).stderr
mean = float(re.search(r"mean_volume: (-?[\d.]+)", vol)[1])
peak = float(re.search(r"max_volume: (-?[\d.]+)", vol)[1])
sil = subprocess.run([FF, "-i", str(mp4), "-vn", "-af", "silencedetect=n=-45dB:d=4", "-f", "null", "-"], capture_output=True, text=True).stderr
silences = re.findall(r"silence_duration: ([\d.]+)", sil)

checks = {
    "codec h264": video and video[1] == "h264",
    "1920x1080": video and (video[2], video[3]) == ("1920", "1080"),
    "30 fps": video and abs(float(video[4]) - 30) < 0.01,
    f"duration {seconds:.2f}s ≈ timeline {expected:.2f}s": abs(seconds - expected) < 0.5,
    "aac audio": audio and audio[1] == "aac",
    f"narration level mean {mean} dB / peak {peak} dB": -35 < mean < -10 and peak < 0,
    f"no silence ≥ 4 s ({len(silences)} found)": len(silences) == 0,
}

tiles = []
for ch in tl["chapters"]:
    t = (ch["start"] + ch["end"]) / 2
    png = subprocess.run([FF, "-v", "error", "-ss", f"{t:.2f}", "-i", str(mp4), "-frames:v", "1", "-f", "image2pipe",
                          "-vcodec", "png", "-"], capture_output=True).stdout
    tmp = VIDEO / "out" / "frames" / f"mp4-{ch['id']}.png"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    if not png:
        checks[f"frame decodes at {t:.0f}s ({ch['id']})"] = False
        continue
    tmp.write_bytes(png)
    tiles.append((ch["title"], t, tmp))
sheet = Image.new("RGB", (4 * 480, 3 * 294), (6, 12, 29))
d = ImageDraw.Draw(sheet)
for i, (title, t, p) in enumerate(tiles):
    x, y = (i % 4) * 480, (i // 4) * 294
    sheet.paste(Image.open(p).convert("RGB").resize((480, 270)), (x, y))
    d.text((x + 6, y + 274), f"{title} @ {t:.0f}s", fill=(243, 233, 214))
sheet.save(VIDEO / "out" / "mp4-contact-sheet.jpg", quality=88)

ok = all(checks.values())
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print("size", round(mp4.stat().st_size / 1e6, 1), "MB · contact sheet out/mp4-contact-sheet.jpg")
sys.exit(0 if ok else 1)
