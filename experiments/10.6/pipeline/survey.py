"""Dataset survey pipeline L1 (all classes, train split).
Per class: collect -> adaptive area filter -> cap -> DINOv2 ViT-S/14 -> adaptive-K KMeans
-> contact sheet + medoids + stats.json + embeddings.pt (reusable for L2 deep dive).
Run: uv run python experiments/10.6/pipeline/survey.py
Outputs: experiments/10.6/atlas/<Class>/{contact_sheet.jpg, medoids/, stats.json, embeddings.pt, manifest.json}
"""
import json
import random
import re
import collections
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

BASE = Path("dataset/batch_12.v14i.merged12.yolov11/train")
ATLAS = Path("experiments/10.6/atlas")
CLASSES = ["Coach", "Franchised Bus", "HGV", "LGV", "Light Bus", "MGV",
           "Motorcycle", "PLB GMB", "Private Car", "Taxi", "Van", "Container"]
AREA_STEPS = [0.01, 0.005, 0.0]
MAXN = 4000
SEED = 7
EXPAND = 0.10
TOPK = 9
THUMB = 160
BATCH = 64

random.seed(SEED)
torch.manual_seed(SEED)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def vid_of(src: str) -> str:
    m = re.search(r"(dahua\d+|cam\d+)", src)
    if m and "_frame_" not in src and src.startswith("202"):
        return m.group(1)
    if "_frame_" in src:
        return src.split("_frame_")[0]
    m2 = re.match(r"(ANMR\d+)", src)
    if m2 and "_f" in src:
        return m2.group(1)
    return src[:16]


def collect(cid: int):
    items = []  # (imgpath, xc,yc,w,h)
    for lf in sorted((BASE / "labels").glob("*.txt")):
        img = BASE / "images" / (lf.stem + ".jpg")
        if not img.exists():
            c = list((BASE / "images").glob(lf.stem + "*"))
            if not c:
                continue
            img = c[0]
        for line in open(lf):
            p = line.strip().split()
            if not p or int(float(p[0])) != cid:
                continue
            items.append((str(img), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    return items


def crop_square(img, xc, yc, w, h):
    W, H = img.size
    cx, cy = xc * W, yc * H
    bw, bh = w * W * (1 + EXPAND), h * H * (1 + EXPAND)
    x0, y0 = int(cx - bw / 2), int(cy - bh / 2)
    x1, y1 = int(cx + bw / 2), int(cy + bh / 2)
    crop = img.crop((max(0, x0), max(0, y0), min(W, x1), min(H, y1)))
    s = max(crop.size) if min(crop.size) > 0 else 1
    cv = Image.new("RGB", (s, s), (0, 0, 0))
    cv.paste(crop, ((s - crop.width) // 2, (s - crop.height) // 2))
    return cv.resize((224, 224), Image.BILINEAR)


@torch.no_grad()
def kmeans_torch(X, k, iters=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    N = X.shape[0]
    Xc = X.cpu()
    first = torch.randint(N, (1,), generator=g).item()
    centers = [Xc[first]]
    for _ in range(1, k):
        d2 = torch.cdist(Xc, torch.stack(centers)).amin(dim=1) ** 2
        prob = d2 / d2.sum().clamp_min(1e-12)
        centers.append(Xc[torch.multinomial(prob, 1, generator=g).item()])
    C = torch.stack(centers).to(X.device)
    labels = torch.zeros(N, dtype=torch.long, device=X.device)
    for _ in range(iters):
        d = torch.cdist(X, C)
        nl = d.argmin(dim=1)
        nC = C.clone()
        for ki in range(k):
            m = nl == ki
            if m.sum() > 0:
                nC[ki] = X[m].mean(dim=0)
                nC[ki] /= nC[ki].norm().clamp_min(1e-12)
        if torch.equal(nl, labels) and _ > 0:
            labels, C = nl, nC
            break
        labels, C = nl, nC
    dists = torch.cdist(X, C).gather(1, labels.unsqueeze(1)).squeeze(1)
    return labels.cpu(), dists.cpu()


def pick_k(n):
    if n >= 1500:
        return 5
    if n >= 600:
        return 4
    if n >= 200:
        return 3
    if n >= 60:
        return 2
    return 1


def flag_cluster(n, uv, top1share):
    if n >= 20 and (uv <= 3 or top1share >= 0.6):
        return "artifact:single-source"
    if n >= 20 and top1share >= 0.4:
        return "artifact:source-dominated"
    return "subtype-candidate"


def main():
    ATLAS.mkdir(parents=True, exist_ok=True)
    log = open(ATLAS / "survey_log.txt", "w")
    def say(s):
        print(s, flush=True)
        log.write(s + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    say(f"device={device}")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
    summary = []

    for cid, cname in enumerate(CLASSES):
        say(f"=== [{cid}] {cname} ===")
        items = collect(cid)
        n_all = len(items)
        amin, qual = AREA_STEPS[-1], items
        for a in AREA_STEPS:
            q = [it for it in items if it[3] * it[4] >= a]
            if len(q) >= 150 or a == AREA_STEPS[-1]:
                amin, qual = a, q
                break
        say(f"boxes={n_all} qualified(area>={amin})={len(qual)} ({len(qual)/max(1,n_all)*100:.1f}%)")
        rng = random.Random(SEED * 1000 + cid)
        used = rng.sample(qual, min(MAXN, len(qual))) if len(qual) > MAXN else list(qual)
        n = len(used)
        K = pick_k(n)
        say(f"embed n={n} K={K}")
        if n == 0:
            summary.append({"cid": cid, "name": cname, "boxes": 0, "qualified": 0, "k": 0, "flag": "empty"})
            continue

        # group by image: open once
        byimg = collections.defaultdict(list)
        for j, it in enumerate(used):
            byimg[it[0]].append((j,) + it[1:])
        thumbs, tens = [None] * n, [None] * n
        done = 0
        with torch.no_grad():
            buf, bidx = [], []
            all_t = [None] * n
            for ip, rows in byimg.items():
                img = Image.open(ip).convert("RGB")
                for (j, xc, yc, w, h) in rows:
                    sq = crop_square(img, xc, yc, w, h)
                    thumbs[j] = sq
                    buf.append(tf(sq))
                    bidx.append(j)
                    if len(buf) == BATCH:
                        out = model(torch.stack(buf).to(device)).cpu()
                        for jj, t in zip(bidx, out):
                            all_t[jj] = t
                        buf, bidx = [], []
                done += len(rows)
                if done % 1000 < len(rows):
                    say(f"  crop+embed {min(done,n)}/{n}")
            if buf:
                out = model(torch.stack(buf).to(device)).cpu()
                for jj, t in zip(bidx, out):
                    all_t[jj] = t
        F = torch.stack(all_t)
        Xn = F / F.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if K > 1:
            labels, dists = kmeans_torch(Xn.to(device), K, seed=SEED)
        else:
            labels = torch.zeros(n, dtype=torch.long)
            dists = torch.cdist(Xn, Xn.mean(dim=0, keepdim=True)).squeeze(1)

        srcs = [Path(ip).name for ip, *_ in used]
        vids = [vid_of(s) for s in srcs]
        areas = [w * h for _, _, _, w, h in used]
        out = ATLAS / cname
        (out / "medoids").mkdir(parents=True, exist_ok=True)
        torch.save({"feats": F}, out / "embeddings.pt")
        json.dump([{"src": s, "box": [used[j][1], used[j][2], used[j][3], used[j][4]]}
                   for j, s in enumerate(srcs)], open(out / "manifest.json", "w"))

        LABELW = 200
        rows = K
        sheet = Image.new("RGB", (LABELW + TOPK * THUMB, rows * THUMB), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        stat = {"class": cname, "cid": cid, "boxes_train": n_all, "qualified": len(qual),
                "area_min": amin, "embedded": n, "k": K, "median_area": sorted(areas)[n // 2],
                "videos": len(set(vids)), "clusters": []}
        for sk in range(rows):
            members = [j for j in range(n) if int(labels[j]) == sk]
            members.sort(key=lambda j: float(dists[j]))
            pick = members[:TOPK]
            cnt = collections.Counter(vids[j] for j in members)
            uv = len(cnt)
            top1 = cnt.most_common(1)[0][1] / max(1, len(members))
            fl = flag_cluster(len(members), uv, top1)
            say(f"  C{sk}: n={len(members)} videos={uv} top1={top1*100:.0f}% -> {fl}")
            dr.rectangle([0, sk * THUMB, LABELW, (sk + 1) * THUMB], fill=(255, 230, 230) if fl.startswith("artifact") else (240, 240, 240))
            dr.text((10, sk * THUMB + THUMB // 2 - 30), f"C{sk} n={len(members)}\nv={uv}\n{fl.split(':')[0]}", fill=(0, 0, 0))
            for c, j in enumerate(pick):
                sheet.paste(thumbs[j].resize((THUMB, THUMB), Image.BILINEAR), (LABELW + c * THUMB, sk * THUMB))
                thumbs[j].save(out / "medoids" / f"C{sk}_rank{c:02d}_{srcs[j]}")
            stat["clusters"].append({"s": sk, "n": len(members), "videos": uv,
                                     "top1_share": round(top1, 3), "flag": fl,
                                     "top_videos": [[a, int(b)] for a, b in cnt.most_common(3)]})
        sheet.save(out / "contact_sheet.jpg", quality=90)
        json.dump(stat, open(out / "stats.json", "w"), indent=1)
        summary.append({"cid": cid, "name": cname, "boxes": n_all, "qualified": len(qual),
                        "embedded": n, "k": K,
                        "artifacts": sum(1 for c in stat["clusters"] if c["flag"].startswith("artifact"))})
        del F, Xn, thumbs
    json.dump(summary, open(ATLAS / "summary.json", "w"), indent=1)
    say("survey done")
    log.close()


if __name__ == "__main__":
    main()
