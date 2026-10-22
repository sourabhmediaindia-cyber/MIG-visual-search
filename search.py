#!/usr/bin/env python3
"""Search the CLIP index by text or by example image.

Usage:
    python search.py --text "Abu Dhabi" -k 20
    python search.py --text "Sheikh Zayed Grand Mosque" --html results.html
    python search.py --image some_photo.jpg -k 20 --html results.html
"""
import argparse
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import open_clip


def normalize_query_text(value):
    text = str(value).lower()
    text = re.sub(r"[_\\/.-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"[^a-z0-9 ]+", " ", text)


def query_tokens(value):
    tokens = normalize_query_text(value).split()
    generic = {"a", "an", "and", "for", "from", "in", "of", "on", "the", "to", "with", "img", "image", "images", "photo", "photos", "pic", "pics", "picture", "pictures", "dsc", "jpeg", "jpg", "png", "webp", "bmp", "tif", "tiff", "folder", "media"}
    return [token for token in tokens if len(token) > 1 and token not in generic]


def lexical_match_score(query, path):
    if not query:
        return 0.0
    q_tokens = set(query_tokens(query))
    if not q_tokens:
        return 0.0
    text = normalize_query_text(path)
    path_tokens = set(query_tokens(path))
    if not path_tokens:
        return 0.0
    overlap = q_tokens & path_tokens
    if not overlap:
        return 0.0
    exact_bonus = 1.0 if normalize_query_text(query) in text else 0.0
    overlap_fraction = len(overlap) / len(q_tokens)
    return min(1.0, 0.7 * overlap_fraction + 0.3 * exact_bonus)


def text_query_variants(text, prompt_template="a photo of {}"):
    raw = (text or "").strip()
    if not raw:
        return [""]
    cleaned = normalize_query_text(raw)
    variants = (
        raw,
        prompt_template.format(cleaned),
        f"an image of {cleaned}",
        f"a photograph of {cleaned}",
    )
    return list(dict.fromkeys(variant for variant in variants if variant.strip()))


def build_text_embedding(model, tokenizer, text, prompt_template="a photo of {}"):
    variants = text_query_variants(text, prompt_template)
    q = model.encode_text(tokenizer(variants))
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = q.mean(dim=0, keepdim=True)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--image")
    ap.add_argument("--index", default="index_data")
    ap.add_argument("-k", type=int, default=20)
    ap.add_argument("--prompt", default="a photo of {}", help="template for text queries")
    ap.add_argument(
        "--min-score",
        type=float,
        default=0.22,
        help="minimum CLIP similarity for text results (default: 0.28)",
    )
    ap.add_argument("--html", help="write a results page with thumbnails (open in browser)")
    args = ap.parse_args()
    if args.k < 1:
        ap.error("-k must be at least 1")
    if not 0.0 <= args.min_score <= 1.0:
        ap.error("--min-score must be between 0 and 1")

    idx = Path(args.index)
    meta = json.loads((idx / "meta.json").read_text(encoding="utf-8"))
    paths = json.loads((idx / "paths.json").read_text(encoding="utf-8"))
    vecs = np.load(idx / "vectors.npy")
    labels = json.loads((idx / "labels.json").read_text(encoding="utf-8")) if (idx / "labels.json").exists() else [[p] for p in paths]

    model, _, preprocess = open_clip.create_model_and_transforms(meta["model"], pretrained=meta["pretrained"])
    model.eval()

    with torch.no_grad():
        if args.text:
            tok = open_clip.get_tokenizer(meta["model"])
            q = build_text_embedding(model, tok, args.text, args.prompt)
            q = q / q.norm(dim=-1, keepdim=True)
        else:
            im = Image.open(args.image).convert("RGB")
            q = model.encode_image(preprocess(im).unsqueeze(0))
            q = q / q.norm(dim=-1, keepdim=True)
    q = q[0].numpy()

    t0 = time.time()
    clip_scores = vecs @ q
    lexical_scores = np.zeros_like(clip_scores, dtype=np.float32)
    if args.text:
        query = args.text
        lexical_scores = np.array([
            lexical_match_score(query, " ".join(labels[i]) if i < len(labels) else paths[i])
            for i in range(len(paths))
        ], dtype=np.float32)
        mix_weight = 0.18
        final_scores = clip_scores + mix_weight * lexical_scores
    else:
        final_scores = clip_scores
    if args.text:
        eligible = np.flatnonzero(clip_scores >= args.min_score)
    else:
        eligible = np.arange(len(paths))
    top = eligible[np.argsort(-final_scores[eligible], kind="stable")[:args.k]]
    dt = (time.time() - t0) * 1000

    print(f"Found {len(top)} matching images out of {len(paths)} (search took {dt:.0f} ms):")
    if args.text and not len(top):
        print(f"No images met the minimum CLIP similarity of {args.min_score:.3f}.")
    for rank, i in enumerate(top, 1):
        print(f"{rank:>3}. {clip_scores[i]:.3f}  {paths[i]}")

    if args.html:
        label = args.text or args.image
        total = len(top)
        cells = []
        for rank, i in enumerate(top, 1):
            uri = Path(paths[i]).resolve().as_uri()
            rel = Path(paths[i])
            meta_path = f"{rel.parent.name}/{rel.name}"
            cells.append(
                "<article class=\"card\">"
                f"<a href=\"{uri}\" target=\"_blank\" rel=\"noopener\">"
                "<div class=\"thumb-wrap\">"
                f"<img src=\"{uri}\" alt=\"{html.escape(meta_path, quote=True)}\" loading=\"lazy\">"
                "</div>"
                "</a>"
                "<div class=\"card-body\">"
                "<div class=\"meta-row\">"
                f"<span class=\"rank\">#{rank}</span>"
                f"<span class=\"score\">{clip_scores[i]:.3f}</span>"
                "</div>"
                f"<div class=\"path\">{html.escape(meta_path)}</div>"
                "</div>"
                "</article>"
            )
        page = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Visual Search Results</title>
  <style>
    :root {{
      --bg: #07111f;
      --bg-soft: #101c2d;
      --panel: rgba(16, 28, 45, 0.82);
      --panel-strong: rgba(19, 35, 56, 0.96);
      --border: rgba(148, 163, 184, 0.18);
      --text: #edf4ff;
      --muted: #a7b7d1;
      --accent: #68e0c1;
      --accent-2: #7aa3ff;
      --shadow: rgba(8, 15, 30, 0.5);
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; }}
    body {{
      font-family: Inter, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top, rgba(122, 163, 255, 0.18), transparent 28%),
        linear-gradient(180deg, #060d18 0%, #0b1525 100%);
      color: var(--text);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 32px 18px 64px;
    }}
    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 26px;
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 18px;
      background: rgba(13, 23, 37, 0.76);
      box-shadow: 0 18px 44px var(--shadow);
      backdrop-filter: blur(12px);
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: #dfe9ff;
      font-size: 0.77rem;
    }}
    .brand-mark {{
      width: 12px;
      height: 12px;
      border-radius: 50%;
      background: linear-gradient(135deg, var(--accent), var(--accent-2));
      box-shadow: 0 0 18px rgba(104, 224, 193, 0.8);
    }}
    .query-wrap {{
      display: flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .query-pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(104, 224, 193, 0.12);
      border: 1px solid rgba(104, 224, 193, 0.22);
      color: #d9fff2;
      font-size: 0.75rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .query-text {{
      font-size: clamp(1.2rem, 2.2vw, 2rem);
      font-weight: 700;
      color: var(--text);
      max-width: 680px;
      overflow-wrap: anywhere;
    }}
    .results-meta {{
      font-size: 0.8rem;
      color: var(--muted);
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 22px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 18px 44px rgba(7, 12, 20, 0.38);
      transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }}
    .card:hover {{
      transform: translateY(-4px);
      border-color: rgba(122, 163, 255, 0.32);
      box-shadow: 0 22px 50px rgba(9, 17, 29, 0.52);
    }}
    .thumb-wrap {{
      position: relative;
      aspect-ratio: 4 / 3;
      background: rgba(148, 163, 184, 0.12);
      overflow: hidden;
    }}
    .thumb-wrap img {{
      width: 100%;
      height: 100%;
      display: block;
      object-fit: cover;
      transition: transform 0.35s ease;
    }}
    .card:hover .thumb-wrap img {{
      transform: scale(1.04);
    }}
    .card-body {{
      padding: 14px 14px 16px;
    }}
    .meta-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 8px;
    }}
    .rank {{
      font-size: 0.72rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    .score {{
      display: inline-flex;
      align-items: center;
      padding: 5px 8px;
      border-radius: 999px;
      background: rgba(104, 224, 193, 0.12);
      border: 1px solid rgba(104, 224, 193, 0.2);
      color: #aef7df;
      font-size: 0.76rem;
      font-weight: 700;
    }}
    .path {{
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.4;
      word-break: break-word;
    }}
    @media (max-width: 640px) {{
      .page {{ padding: 22px 14px 40px; }}
      .topbar {{ padding: 16px; }}
      .grid {{ gap: 16px; }}
      .query-wrap {{ align-items: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class=\"page\">
    <header class=\"topbar\">
      <div class=\"brand\"><span class=\"brand-mark\"></span>  <img src="https://mediaindia.eu/wp-content/themes/mig_theme/assets/images/mig-logo-200x140.png" style="width:90px"><h2><b>MEDIA Hub</b></h2></div>
      <div class=\"query-wrap\">
        <span class=\"query-pill\">Query</span>
        <div class=\"query-text\">{html.escape(str(label), quote=True)}</div>
      </div>
      <div class=\"results-meta\">{total} results</div>
    </header>
    <main class=\"grid\">{''.join(cells) if cells else '<p class=\"path\">No images met the minimum similarity threshold.</p>'}</main>
  </div>
</body>
</html>
"""
        Path(args.html).write_text(page, encoding="utf-8")
        print(f"Wrote {args.html}")


if __name__ == "__main__":
    main()
