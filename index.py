#!/usr/bin/env python3
"""Build a CLIP image index for a folder (recursive).

Usage:
    python index.py "D:\\media\\abu_dhabi_test" --out index_data
    python index.py "\\\\server\\share\\photos" --out index_data --limit 200

Re-running on the same --out folder only processes NEW files (resume-friendly).
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 300_000_000

EXTS = {".jpg", ".jpeg", ".jfif", ".png", ".webp", ".bmp", ".tif", ".tiff"}
GENERIC_TOKENS = {"img", "image", "images", "photo", "photos", "pic", "pics", "dsc", "jpeg", "jpg", "png", "webp", "bmp", "tif", "tiff", "camera", "shot", "picture", "media", "folder"}


def normalize_tokens(value):
    text = str(value).lower()
    text = re.sub(r"[_\\/.-]+", " ", text)
    text = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", text)
    tokens = re.split(r"[^a-z0-9]+", text)
    cleaned = []
    for token in tokens:
        if not token:
            continue
        if len(token) == 1 and not token.isdigit():
            continue
        if token in GENERIC_TOKENS:
            continue
        cleaned.append(token)
    return cleaned


def path_terms(path):
    p = Path(path)
    text = " ".join([p.stem, p.parent.name])
    return normalize_tokens(text)


class ImageDS(torch.utils.data.Dataset):
    def __init__(self, paths, preprocess, size=224):
        self.paths = paths
        self.preprocess = preprocess
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i])
            # JPEG draft mode decodes a downscaled version: much faster on CPU
            im.draft("RGB", (self.size * 2, self.size * 2))
            im = im.convert("RGB")
            return i, self.preprocess(im)
        except Exception:
            return i, None


def collate(batch):
    batch = [b for b in batch if b[1] is not None]
    if not batch:
        return None
    idx = torch.tensor([b[0] for b in batch])
    x = torch.stack([b[1] for b in batch])
    return idx, x


def scan(folder):
    found = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                found.append(str(Path(root, f)))
    found.sort()
    return found


def save(out, paths, vecs, meta, labels=None):
    out.mkdir(parents=True, exist_ok=True)
    tmp_v = out / "vectors.tmp.npy"
    np.save(tmp_v, vecs.astype(np.float32))
    os.replace(tmp_v, out / "vectors.npy")
    tmp_p = out / "paths.tmp.json"
    tmp_p.write_text(json.dumps(paths), encoding="utf-8")
    os.replace(tmp_p, out / "paths.json")
    if labels is None:
        labels = [path_terms(p) for p in paths]
    tmp_l = out / "labels.tmp.json"
    tmp_l.write_text(json.dumps(labels), encoding="utf-8")
    os.replace(tmp_l, out / "labels.json")
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="index_data")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="laion2b_s34b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=0, help="only index first N new images (for testing)")
    ap.add_argument("--checkpoint", type=int, default=2000, help="save every N images")
    args = ap.parse_args()

    out = Path(args.out)
    meta = {"model": args.model, "pretrained": args.pretrained}

    paths, vecs, labels = [], np.zeros((0, 0), dtype=np.float32), []
    if (out / "paths.json").exists() and (out / "vectors.npy").exists():
        old_meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        if old_meta != meta:
            sys.exit(f"Existing index in {out} was built with {old_meta}; use a new --out or the same model.")
        paths = json.loads((out / "paths.json").read_text(encoding="utf-8"))
        vecs = np.load(out / "vectors.npy")
        labels = json.loads((out / "labels.json").read_text(encoding="utf-8")) if (out / "labels.json").exists() else [path_terms(p) for p in paths]
        print(f"Resuming: {len(paths)} images already indexed")

    print("Scanning folder...")
    # compare normalized paths so C:\A\x.jpg, c:/a/x.jpg etc. count as the same file
    def norm(p):
        return os.path.normcase(os.path.abspath(p))

    done = {norm(p) for p in paths}
    new_paths = [p for p in scan(os.path.abspath(args.folder)) if norm(p) not in done]
    if args.limit:
        new_paths = new_paths[: args.limit]
    print(f"{len(new_paths)} new images to index")
    if not new_paths:
        return

    torch.set_num_threads(os.cpu_count() or 4)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model.eval()

    dl = torch.utils.data.DataLoader(
        ImageDS(new_paths, preprocess),
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate,
    )

    feats_buf, paths_buf, labels_buf = [], [], []
    processed = 0
    since_ckpt = 0
    t0 = time.time()

    def flush():
        nonlocal paths, vecs, feats_buf, paths_buf, labels, labels_buf
        if not feats_buf:
            return
        new_v = np.concatenate(feats_buf, axis=0)
        vecs = new_v if vecs.size == 0 else np.concatenate([vecs, new_v], axis=0)
        paths = paths + paths_buf
        labels = labels + labels_buf
        feats_buf, paths_buf, labels_buf = [], [], []
        save(out, paths, vecs, meta, labels)

    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            idx, x = batch
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats_buf.append(f.cpu().numpy())
            batch_paths = [new_paths[i] for i in idx.tolist()]
            paths_buf.extend(batch_paths)
            labels_buf.extend(path_terms(p) for p in batch_paths)
            processed += len(idx)
            since_ckpt += len(idx)
            if since_ckpt >= args.checkpoint:
                flush()
                since_ckpt = 0
            rate = processed / (time.time() - t0)
            print(f"\r{processed}/{len(new_paths)}  {rate:.1f} img/s", end="", flush=True)

    flush()
    dt = time.time() - t0
    print(f"\nDone: {processed} images in {dt:.0f}s ({processed / max(dt, 1e-9):.1f} img/s). Total indexed: {len(paths)}")
    if processed:
        eta_days = 1_000_000 / (processed / dt) / 86400
        print(f"Estimated time for 10 lakh images at this speed: {eta_days:.1f} days")


if __name__ == "__main__":
    main()