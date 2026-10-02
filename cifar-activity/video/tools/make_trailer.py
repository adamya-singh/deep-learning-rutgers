"""Assemble out/cifar-journey-trailer.mp4 from narrated excerpts of the real composition.

    .venv-tools/bin/python tools/make_trailer.py --plan   # print the render_clips.sh spec for the excerpts
    tools/render_clips.sh $(.venv-tools/bin/python tools/make_trailer.py --plan)
    .venv-tools/bin/python tools/make_trailer.py          # out/qa/trailer-*.mp4 → out/cifar-journey-trailer.mp4

Excerpts are whole narration segments (listed in PIECES), so frame ranges follow
src/data/timeline.json automatically when the narration changes. Each piece gets
a short fade in/out; audio and video are re-encoded together.
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

VIDEO = Path(__file__).resolve().parents[1]
OUT = VIDEO / "out"
FF = imageio_ffmpeg.get_ffmpeg_exe()
# (first segment, last segment) per excerpt, in film order
PIECES = [("task-1", "task-2"), ("cnn-3", "cnn-3"), ("learn-7", "learn-7"), ("depth-3", "depth-3"),
          ("aug2-2", "aug2-2"), ("resnet-3", "resnet-3"), ("end-1", "end-1")]


def plan():
    tl = json.loads((VIDEO / "src" / "data" / "timeline.json").read_text())
    segs = {s["id"]: s for c in tl["chapters"] for s in c["segments"]}
    specs = []
    for i, (a, b) in enumerate(PIECES, 1):
        start = max(0, round((segs[a]["start"] - 0.25) * tl["fps"]))
        end = round((segs[b]["start"] + segs[b]["duration"] + 0.2) * tl["fps"])
        specs.append(f"trailer-t{i}:{start}-{min(end, tl['durationInFrames'] - 1)}")
    return specs


def duration(p):
    err = subprocess.run([FF, "-i", str(p)], capture_output=True, text=True).stderr
    h, m, s = err.split("Duration: ")[1].split(",")[0].split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


if "--plan" in sys.argv:
    print(" ".join(plan()))
    sys.exit(0)

pieces = [OUT / "qa" / f"{spec.split(':')[0]}.mp4" for spec in plan()]
missing = [p.name for p in pieces if not p.exists()]
if missing:
    sys.exit(f"missing excerpts {missing}; run: tools/render_clips.sh $(.venv-tools/bin/python tools/make_trailer.py --plan)")
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
