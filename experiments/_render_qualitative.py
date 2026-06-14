"""F10: side-by-side qualitative clips B0 vs B4 on occlusion-heavy val frames.

Reads the MOT-format tracker outputs (`dancetrack<seq>.txt`) from both runs,
overlays them on val frames, and writes one mp4 per chosen sequence.

Assumes DanceTrack val images at:
    <data_root>/val/dancetrack<seq>/img1/<frame>.jpg

Sequences worth showing (occlusion-heavy on val, from `pedestrian_detailed.csv`):
    dancetrack0014, dancetrack0026, dancetrack0034, dancetrack0058, dancetrack0073
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2  # opencv-python
import numpy as np
from tqdm import tqdm

PALETTE = [(31,119,180),(255,127,14),(44,160,44),(214,39,40),(148,103,189),
           (140,86,75),(227,119,194),(127,127,127),(188,189,34),(23,190,207)]


def load_mot(path: Path) -> dict[int, list[tuple]]:
    """frame -> [(id, x, y, w, h)]."""
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        f, tid, x, y, w, h, *_ = line.split(",")
        out.setdefault(int(f), []).append((int(tid), float(x), float(y), float(w), float(h)))
    return out


def draw(img, dets, tag):
    for tid, x, y, w, h in dets:
        c = PALETTE[tid % len(PALETTE)]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
        cv2.putText(img, f"{tid}", (int(x), int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.putText(img, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-tracker", type=Path, required=True,
                    help="Directory of B0 dancetrack<seq>.txt files.")
    ap.add_argument("--b4-tracker", type=Path, required=True)
    ap.add_argument("--data-root",  type=Path, required=True, help="DanceTrack root.")
    ap.add_argument("--seqs", nargs="+",
                    default=["dancetrack0014", "dancetrack0026", "dancetrack0034"])
    ap.add_argument("--out", type=Path, default=Path("/workspace/persistent/results/_figures/F10_qualitative"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for seq in tqdm(args.seqs):
        b0 = load_mot(args.b0_tracker / f"{seq}.txt")
        b4 = load_mot(args.b4_tracker / f"{seq}.txt")
        img_dir = args.data_root / "val" / seq / "img1"
        frames = sorted(img_dir.glob("*.jpg"))
        if not frames:
            print(f"[skip] no frames at {img_dir}")
            continue
        h, w = cv2.imread(str(frames[0])).shape[:2]
        out_mp4 = args.out / f"{seq}_b0_vs_b4.mp4"
        vw = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 20, (2 * w, h))
        for i, fp in enumerate(frames, 1):
            img = cv2.imread(str(fp))
            left = draw(img.copy(), b0.get(i, []), "B0 (released)")
            right = draw(img.copy(), b4.get(i, []), "B4 (signal + symmetric)")
            vw.write(np.hstack([left, right]))
        vw.release()
        print(f"Wrote {out_mp4}")


if __name__ == "__main__":
    main()
