"""Assemble out/cifar-journey-trailer.mp4 from out/qa/trailer-*.mp4 (rendered by
tools/render_clips.sh from the real composition). Each piece gets a short
fade in/out; audio and video are re-encoded together.

    .venv-tools/bin/python tools/make_trailer.py
"""
import subprocess
from pathlib import Path

import imageio_ffmpeg

OUT = Path(__file__).resolve().parents[1] / "out"
FF = imageio_ffmpeg.get_ffmpeg_exe()
pieces = sorted((OUT / "qa").glob("trailer-*.mp4"))


def duration(p):
    err = subprocess.run([FF, "-i", str(p)], capture_output=True, text=True).stderr
    h, m, s = err.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


args, filters = [FF, "-v", "error", "-y"], []
for i, p in enumerate(pieces):
    d = duration(p)
    args += ["-i", str(p)]
    filters.append(f"[{i}:v]fade=t=in:st=0:d=0.35,fade=t=out:st={d - 0.4:.3f}:d=0.4[v{i}]")
    filters.append(f"[{i}:a]afade=t=in:st=0:d=0.25,afade=t=out:st={d - 0.4:.3f}:d=0.4[a{i}]")
concat = "".join(f"[v{i}][a{i}]" for i in range(len(pieces)))
filters.append(f"{concat}concat=n={len(pieces)}:v=1:a=1[v][a]")
args += ["-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-crf", "20",
         "-preset", "medium", "-threads", "4", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-movflags", "+faststart", str(OUT / "cifar-journey-trailer.mp4")]
subprocess.run(args, check=True)
print("trailer:", OUT / "cifar-journey-trailer.mp4", f"{sum(duration(p) for p in pieces):.1f}s from {len(pieces)} pieces")
