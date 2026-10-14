"""Atlas overview figure: one row per class with L1 subtype proportions + stats + medoid thumbs.
Run: uv run python experiments/10.6/pipeline/atlas_overview.py
Output: experiments/10.6/atlas/OVERVIEW.jpg
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ATLAS = Path("experiments/10.6/atlas")
ORDER = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
         "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container"]
CAND = ["#4C78A8", "#54A24B", "#8EC0E8", "#7AC36A", "#B8D98A"]
ART = "#E45756"

def font(sz, bold=False):
    p = Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), sz)
    return ImageFont.load_default(size=sz)

F_TITLE = font(44, True)
F_NAME = font(32, True)
F_TXT = font(26)
F_SMALL = font(22)
F_TINY = font(20)


def main():
    stats = {c: json.load(open(ATLAS / c / "stats.json")) for c in ORDER}
    summary = {s["name"]: s for s in json.load(open(ATLAS / "summary.json"))}
    total_boxes = sum(s["boxes"] for s in summary.values())
    total_qual = sum(s["qualified"] for s in summary.values())

    W = 2240
    LEFT, BARW, GAP = 470, 780, 50
    TH = 150
    ROWH = 235
    HEADH, FOOTH = 250, 250
    H = HEADH + ROWH * len(ORDER) + FOOTH
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    d.text((40, 30), "Dataset Visual Atlas  |  train split  |  DINOv2 ViT-S/14, area>=0.01", font=F_TITLE, fill=(20, 20, 20))
    d.text((40, 100), f"49,719 boxes total  |  {total_qual} qualified ({total_qual/total_boxes*100:.0f}%)  |  "
           "L1 KMeans per class  |  red = single-source artifact, blue/green = cross-video subtype",
           font=F_TXT, fill=(80, 80, 80))
    d.text((40, 145), "bar = cluster share of embedded crops  |  v = distinct videos  |  top1 = largest single-video share",
           font=F_TXT, fill=(80, 80, 80))
    d.line([(0, HEADH - 10), (W, HEADH - 10)], fill=(0, 0, 0), width=3)

    thx = LEFT + GAP + BARW + GAP
    for ri, c in enumerate(ORDER):
        st = stats[c]
        y = HEADH + ri * ROWH
        if ri % 2:
            d.rectangle([0, y, W, y + ROWH], fill=(247, 247, 247))
        # left stats
        d.text((40, y + 8), c, font=F_NAME, fill=(0, 0, 0))
        qp = st["qualified"] / max(1, st["boxes_train"]) * 100
        arts = sum(1 for k in st["clusters"] if k["flag"].startswith("artifact"))
        d.text((40, y + 52), f"boxes {st['boxes_train']}", font=F_TXT, fill=(40, 40, 40))
        d.text((40, y + 86), f"qual {st['qualified']} ({qp:.0f}%)", font=F_TXT, fill=(40, 40, 40))
        d.text((40, y + 120), f"videos {st['videos']}  K={st['k']}", font=F_TXT, fill=(40, 40, 40))
        d.text((40, y + 154), f"artifacts {arts}", font=F_TXT, fill=(ART if arts else (40, 40, 40)))
        d.line([(LEFT - 15, y + 5), (LEFT - 15, y + ROWH - 5)], fill=(200, 200, 200), width=2)
        # stacked bar
        bx, by = LEFT + GAP, y + 18
        n = st["embedded"]
        x = bx
        for ki, k in enumerate(st["clusters"]):
            wseg = max(8, BARW * k["n"] / max(1, n))
            col = ART if k["flag"].startswith("artifact") else CAND[ki % len(CAND)]
            d.rectangle([x, by, min(x + wseg, bx + BARW), by + 44], fill=col, outline=(255, 255, 255), width=2)
            pct = k["n"] / max(1, n) * 100
            if wseg > 90:
                d.text((x + 8, by + 8), f"C{k['s']} {pct:.0f}%", font=F_SMALL, fill=(255, 255, 255))
            x += wseg
        # cluster annotations
        ay = by + 54
        for ki, k in enumerate(st["clusters"]):
            col = ART if k["flag"].startswith("artifact") else CAND[ki % len(CAND)]
            d.rectangle([bx, ay, bx + 16, ay + 16], fill=col)
            mark = "  ARTIFACT" if k["flag"].startswith("artifact") else ""
            d.text((bx + 24, ay - 4),
                   f"C{k['s']}: n={k['n']} v={k['videos']} top1={k['top1_share']*100:.0f}%{mark}",
                   font=F_TINY, fill=(ART if mark else (30, 30, 30)))
            ay += 26
        # medoid thumbs
        for ki, k in enumerate(st["clusters"]):
            f = sorted((ATLAS / c / "medoids").glob(f"C{k['s']}_rank00_*"))
            if not f:
                continue
            t = Image.open(f[0]).convert("RGB").resize((TH, TH), Image.BILINEAR)
            px = thx + ki * (TH + 10)
            img.paste(t, (px, y + 40))
            bw = 6 if k["flag"].startswith("artifact") else 2
            d.rectangle([px, y + 40, px + TH, y + 40 + TH],
                        outline=(ART if k["flag"].startswith("artifact") else (120, 120, 120)), width=bw)
            d.text((px + 4, y + 40 + TH + 2), f"C{k['s']}", font=F_SMALL,
                   fill=(ART if k["flag"].startswith("artifact") else (0, 0, 0)))
        d.line([(0, y + ROWH), (W, y + ROWH)], fill=(220, 220, 220), width=1)

    fy = HEADH + ROWH * len(ORDER) + 15
    d.text((40, fy), "How to read", font=F_NAME, fill=(0, 0, 0))
    d.text((40, fy + 45), "subtype-candidate: many videos, low top1 share = stable visual subtype worth naming (e.g. Taxi side/front/rear/top).",
           font=F_TXT, fill=(40, 40, 40))
    d.text((40, fy + 85), "artifact: few videos or top1>=40% = camera/scene/temporal duplication (e.g. PrivateCar C3: 206/232 from one video).",
           font=F_TXT, fill=(40, 40, 40))
    d.text((40, fy + 125), "Confirmed label issue: Motorcycle C0 (n=71) = pedestrians, box verified on source image.  "
           "Source-poor class: Container = 12 videos total.",
           font=F_TXT, fill=(40, 40, 40))
    d.text((40, fy + 165), "Tiny-object reality: only 18% of PrivateCar / Taxi / 10% of Motorcycle boxes pass area>=0.01.",
           font=F_TXT, fill=(40, 40, 40))

    img.save(ATLAS / "OVERVIEW.jpg", quality=90)
    print("saved", ATLAS / "OVERVIEW.jpg", img.size)


if __name__ == "__main__":
    main()
