"""Synthesize narration and build the audio-driven timeline + captions.

    .venv-tools/bin/python tools/tts.py

Voice: Microsoft Edge neural TTS (edge-tts), en-US-AndrewNeural — a stock
synthetic voice, not a clone of anyone. Each segment is cached by a hash of
(text, voice, rate), so editing one line only re-synthesizes that line.

Outputs:
  public/audio/<segment>-<hash>.mp3     narration clips
  src/data/timeline.json                frame-accurate chapters/segments/captions
  out/cifar-journey.srt, .vtt           captions
"""
import asyncio
import difflib
import math
import hashlib
import json
import re
import subprocess
import wave
import io
from pathlib import Path

import edge_tts
import imageio_ffmpeg

VIDEO = Path(__file__).resolve().parents[1]
VOICE, RATE = "en-US-AndrewNeural", "+15%"
FPS = 30
CHAPTER_LEAD, CHAPTER_TAIL = 0.9, 0.6  # seconds of visual-only time around each chapter's narration
AUDIO = VIDEO / "public" / "audio"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


async def synth(text, path):
    words = []
    comm = edge_tts.Communicate(text, VOICE, rate=RATE, boundary="WordBoundary")
    with open(path, "wb") as f:
        async for ch in comm.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append({"t": ch["offset"] / 1e7, "d": ch["duration"] / 1e7, "w": ch["text"]})
    return words


def duration(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-f", "wav", "-ac", "1", "-ar", "24000", "-"],
                         capture_output=True, check=True).stdout
    with wave.open(io.BytesIO(raw)) as w:
        # ffmpeg writes a placeholder size when piping; compute from byte length instead.
        return (len(raw) - 44) / (w.getsampwidth() * w.getframerate())


def complete(spoken, m):
    """A clip is complete if every spoken word got a boundary and the audio outlasts the last word."""
    words = m["words"]
    if not words:
        return False
    tokens = [t for t in spoken.split() if norm(t)]
    last = words[-1]
    return len(words) >= 0.9 * len(tokens) and m["duration"] >= last["t"] + last["d"] - 0.05 \
        and norm(last["w"])[-3:] == norm(tokens[-1])[-3:]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def time_tokens(text, words, dur):
    """Assign a start time to every whitespace token of the displayed text."""
    tokens = text.split()
    if words and len([t for t in tokens if norm(t)]) == len(words):
        it = iter(words)
        out, last = [], 0.0
        for t in tokens:
            if norm(t):
                last = next(it)["t"]
            out.append((t, last))
        return out
    # Fallback: proportional to character position.
    total = len(text)
    pos, out = 0, []
    for t in tokens:
        i = text.index(t, pos)
        out.append((t, dur * i / total))
        pos = i + len(t)
    return out


def time_tokens_say(text, say, words, dur):
    """Display text differs from the spoken text (pronunciation overrides such as
    CIFAR → SIFAR ten): time the spoken tokens from the word boundaries, then
    carry those times over to the display tokens by sequence alignment."""
    spoken = time_tokens(say, words, dur)
    disp = text.split()
    sm = difflib.SequenceMatcher(a=[norm(t) for t in disp], b=[norm(t) for t, _ in spoken], autojunk=False)
    times = [None] * len(disp)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(i1, i2):
            j = j1 + (k - i1) if tag == "equal" else min(j1, len(spoken) - 1)
            times[k] = spoken[j][1]
    return list(zip(disp, times))


def chunks(timed, end, max_chars=72):
    """Caption chunks: one per sentence, long sentences split into balanced
    pieces at word boundaries (preferring commas), never mid-phrase fragments."""
    sents, cur = [], []
    for tok, t in timed:
        cur.append((tok, t))
        if tok.endswith((".", "?", "!")):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    groups = []
    for sent in sents:
        text = " ".join(x for x, _ in sent)
        k = max(1, math.ceil(len(text) / max_chars))
        if k == 1:
            groups.append(sent)
            continue
        cum, pos = [], 0
        for tok, _ in sent:
            pos += len(tok) + 1
            cum.append(pos)
        cuts = []
        for j in range(1, k):
            target = len(text) * j / k
            best = None
            for idx in range(len(sent) - 1):
                if cuts and idx <= cuts[-1]:
                    continue
                score = abs(cum[idx] - target) - (14 if sent[idx][0].endswith((",", ";", ":")) else 0)
                if best is None or score < best[0]:
                    best = (score, idx)
            cuts.append(best[1])
        prev = 0
        for c in cuts + [len(sent) - 1]:
            groups.append(sent[prev:c + 1])
            prev = c + 1
    out = [{"text": " ".join(x for x, _ in g), "start": g[0][1]} for g in groups if g]
    for n, c in enumerate(out):
        c["end"] = out[n + 1]["start"] if n + 1 < len(out) else end
    return out


def ts(sec, sep):
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


async def main():
    AUDIO.mkdir(parents=True, exist_ok=True)
    script = json.loads((VIDEO / "script" / "narration.json").read_text())
    used = set()
    timeline = {"fps": FPS, "voice": VOICE, "rate": RATE, "snapshot_utc": script["snapshot_utc"], "chapters": []}
    t = 0.0
    captions = []
    for ch in script["chapters"]:
        start = t
        t += CHAPTER_LEAD
        segs = []
        for s in ch["segments"]:
            spoken = s["say"] or s["text"]
            h = hashlib.sha1(f"{VOICE}|{RATE}|{spoken}".encode()).hexdigest()[:10]
            mp3 = AUDIO / f"{s['id']}-{h}.mp3"
            meta = mp3.with_suffix(".json")
            if mp3.exists() and meta.exists() and not complete(spoken, json.loads(meta.read_text())):
                meta.unlink()
            if not (mp3.exists() and meta.exists()):
                for attempt in range(5):
                    try:
                        words = await synth(spoken, mp3)
                        m = {"words": words, "duration": duration(mp3), "spoken": spoken}
                        if complete(spoken, m):
                            break
                        print("incomplete stream, retrying", s["id"])
                    except Exception as e:  # network hiccup
                        print("retry", s["id"], e)
                    await asyncio.sleep(2 + attempt * 3)
                else:
                    raise RuntimeError(f"TTS failed for {s['id']}")
                meta.write_text(json.dumps(m))
            m = json.loads(meta.read_text())
            used.add(mp3.name)
            used.add(meta.name)
            dur = m["duration"]
            timed = time_tokens_say(s["text"], spoken, m["words"], dur) if s["say"] else time_tokens(s["text"], m["words"], dur)
            caps = chunks(timed, dur)
            seg = {"id": s["id"], "audio": f"audio/{mp3.name}", "start": round(t, 3), "duration": round(dur, 3),
                   "text": s["text"], "words": [[tok, round(t + tt, 3)] for tok, tt in timed],
                   "captions": [{"text": c["text"], "start": round(t + c["start"], 3),
                                                    "end": round(t + c["end"], 3)} for c in caps]}
            captions += seg["captions"]
            segs.append(seg)
            # Advance at the last spoken word (clips carry ~0.3 s of trailing silence).
            # Equation pauses (>= 0.9 s) are kept intact; shorter breaths are tightened.
            spoken_end = min(dur, m["words"][-1]["t"] + m["words"][-1]["d"] + 0.12)
            t += spoken_end + (s["pause"] if s["pause"] >= 0.9 else 0.8 * s["pause"])
        t += CHAPTER_TAIL
        timeline["chapters"].append({"id": ch["id"], "title": ch["title"], "start": round(start, 3),
                                     "end": round(t, 3), "segments": segs})
    timeline["duration"] = round(t, 3)
    timeline["durationInFrames"] = int(round(t * FPS))
    (VIDEO / "src" / "data").mkdir(parents=True, exist_ok=True)
    (VIDEO / "src" / "data" / "timeline.json").write_text(json.dumps(timeline, indent=1, ensure_ascii=False))

    out = VIDEO / "out"
    out.mkdir(exist_ok=True)
    srt, vtt = [], ["WEBVTT", ""]
    for i, c in enumerate(captions, 1):
        srt += [str(i), f"{ts(c['start'], ',')} --> {ts(c['end'], ',')}", c["text"], ""]
        vtt += [f"{ts(c['start'], '.')} --> {ts(c['end'], '.')}", c["text"], ""]
    (out / "cifar-journey.srt").write_text("\n".join(srt))
    (out / "cifar-journey.vtt").write_text("\n".join(vtt))
    for f in AUDIO.iterdir():  # drop stale clips from earlier script versions
        if f.name not in used:
            f.unlink()
    for c in timeline["chapters"]:
        print(f"{c['id']:9s} {c['start']:7.1f} → {c['end']:7.1f}  ({c['end'] - c['start']:5.1f}s)")
    print(f"total {t:.1f}s = {t / 60:.2f} min, {timeline['durationInFrames']} frames")


if __name__ == "__main__":
    asyncio.run(main())
