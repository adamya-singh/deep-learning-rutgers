"""Tile rendered stills into contact sheets: .venv-tools/bin/python tools/contact_sheet.py"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "out"
frames = sorted((OUT / "frames").glob("*.jpg"))
TW, TH, COLS, PER = 480, 270, 4, 24
for sheet in range(0, len(frames), PER):
    batch = frames[sheet:sheet + PER]
    rows = (len(batch) + COLS - 1) // COLS
    im = Image.new("RGB", (COLS * TW, rows * (TH + 24)), (6, 12, 29))
    d = ImageDraw.Draw(im)
    for i, f in enumerate(batch):
        t = Image.open(f).resize((TW, TH))
        x, y = (i % COLS) * TW, (i // COLS) * (TH + 24)
        im.paste(t, (x, y))
        d.text((x + 6, y + TH + 4), f.stem, fill=(243, 233, 214))
    im.save(OUT / f"contact-sheet-{sheet // PER + 1}.jpg", quality=88)
print("sheets:", (len(frames) + PER - 1) // PER)
