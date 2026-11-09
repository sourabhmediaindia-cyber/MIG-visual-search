#!/usr/bin/env python3
"""Extract English text for images already present in an image index."""
import argparse
import json
import time
from pathlib import Path

from ocr import create_ocr_reader, extract_ocr_text, save_ocr_texts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="index_data", help="folder containing the image index")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of unprocessed images")
    parser.add_argument("--checkpoint", type=int, default=100, help="save after this many images")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")
    if args.checkpoint < 1:
        parser.error("--checkpoint must be at least 1")

    index_dir = Path(args.index)
    paths = json.loads((index_dir / "paths.json").read_text(encoding="utf-8"))
    text_path = index_dir / "ocr_text.json"
    texts = json.loads(text_path.read_text(encoding="utf-8")) if text_path.exists() else []
    texts = (texts + [None] * len(paths))[:len(paths)]
    pending = [index for index, text in enumerate(texts) if text is None]
    if args.limit:
        pending = pending[:args.limit]
    print(f"{len(pending)} indexed images need OCR.")
    if not pending:
        return

    reader = create_ocr_reader()
    started = time.time()
    for completed, index in enumerate(pending, 1):
        texts[index] = extract_ocr_text(reader, paths[index])
        if completed % args.checkpoint == 0:
            save_ocr_texts(index_dir, texts)
            print(f"OCR processed {completed}/{len(pending)} images.")
    save_ocr_texts(index_dir, texts)
    elapsed = time.time() - started
    print(f"Done: OCR processed {len(pending)} images in {elapsed:.0f}s.")


if __name__ == "__main__":
    main()
