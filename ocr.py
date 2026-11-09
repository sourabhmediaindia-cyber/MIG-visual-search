import json
import os
import sys
from pathlib import Path


def create_ocr_reader():
    try:
        import easyocr
    except ImportError as exc:
        raise RuntimeError(
            "OCR requires EasyOCR. Install project requirements before indexing OCR text."
        ) from exc

    model_directory = os.environ.get("MIG_OCR_MODEL_DIR")
    options = {"gpu": False, "verbose": False}
    if model_directory:
        options["model_storage_directory"] = model_directory
    return easyocr.Reader(["en"], **options)


def extract_ocr_text(reader, image_path):
    try:
        results = reader.readtext(str(image_path), detail=0, paragraph=True)
    except Exception as exc:
        print(f"OCR failed for {image_path}: {exc}", file=sys.stderr)
        return ""
    return " ".join(text.strip() for text in results if text.strip())


def save_ocr_texts(index_dir, texts):
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = index_dir / "ocr_text.tmp.json"
    temporary_path.write_text(
        json.dumps(texts, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, index_dir / "ocr_text.json")
