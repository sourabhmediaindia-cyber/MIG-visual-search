#!/usr/bin/env python3
"""Check how many photos in a folder carry EXIF GPS coordinates.

Usage:
    python check_gps.py "C:\\Users\\GFX3\\Desktop\\test folder"
"""
import os
import sys

from PIL import Image

EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".webp"}


def to_deg(dms, ref):
    d, m, s = (float(x) for x in dms)
    deg = d + m / 60.0 + s / 3600.0
    return -deg if ref in ("S", "W") else deg


def read_gps(path):
    try:
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)  # GPSInfo
    except Exception:
        return None
    if not gps or 2 not in gps or 4 not in gps:
        return None
    try:
        return to_deg(gps[2], gps.get(1, "N")), to_deg(gps[4], gps.get(3, "E"))
    except Exception:
        return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    total = with_gps = 0
    samples = []
    for root, _d, files in os.walk(sys.argv[1]):
        for f in files:
            if os.path.splitext(f)[1].lower() not in EXTS:
                continue
            total += 1
            g = read_gps(os.path.join(root, f))
            if g:
                with_gps += 1
                if len(samples) < 10:
                    samples.append((f, g))
    print(f"{with_gps} of {total} photos have GPS coordinates")
    for f, (lat, lon) in samples:
        print(f"  {f}: {lat:.5f}, {lon:.5f}")
    if total and not with_gps:
        print("No GPS found: city filtering will have to rely on the model (less reliable).")


if __name__ == "__main__":
    main()