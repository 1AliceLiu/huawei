"""Spot-check Motorcycle C0 medoids against original images (draw YOLO box)."""
from pathlib import Path
from PIL import Image, ImageDraw
import json

BASE = Path("dataset/batch_12.v14i.merged12.yolov11/train")
ATLAS = Path("experiments/10.6/atlas/Motorcycle")
man = json.load(open(ATLAS / "manifest.json"))

meds = sorted((ATLAS / "medoids").glob("C0_rank*.jpg"))[:3]
print([m.name for m in meds])
out = []
for m in meds:
    src = m.name.split("C0_rank")[1][3:]  # rankXX_<src>
    # src may contain underscores; manifest src is filename: find entry
    cands = [e for e in man if e["src"] == src]
    print(m.name, "->", src, "matches:", len(cands))
    # find original image
    stem = Path(src).stem
    imgp = BASE / "images" / (stem + ".jpg")
    if not imgp.exists():
        c = list((BASE / "images").glob(stem + "*"))
        imgp = c[0] if c else None
    print("img:", imgp)
    img = Image.open(imgp).convert("RGB")
    W, H = img.size
    d = ImageDraw.Draw(img)
    for e in cands:
        xc, yc, w, h = e["box"]
        d.rectangle([ (xc-w/2)*W, (yc-h/2)*H, (xc+w/2)*W, (yc+h/2)*H ], outline=(255,0,0), width=3)
    img.save(ATLAS / f"spotcheck_{m.stem}.jpg", quality=90)
print("saved spotchecks")
