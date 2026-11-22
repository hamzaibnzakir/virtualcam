"""Turns folders of face photos into training shards for distillation.

For every image: find the largest face, skip it if it looks under 25 (age model) or is unclear,
then store
  - the 128x128 crop aligned the way the swap model (and the VirtualCam app) aligns it, uint8 RGB;
  - its ArcFace identity embedding (float16, 512).
Shards: <out>/shard_00000.npz with arrays `crops` (N,128,128,3) and `emb` (N,512).

  python prepare_data.py --images /data/sfhq1 /data/sfhq2 --models /models --out /data/prepared --limit 120000
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from faces import AgeGender, Detector, Recognizer, align  # noqa: E402

EXTS = {'.jpg', '.jpeg', '.png', '.webp'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', nargs='+', required=True, help='folders of face photos (searched recursively)')
    ap.add_argument('--models', required=True, help='folder with det_10g.onnx, w600k_r50.onnx, genderage.onnx')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='stop after this many kept faces (0 = all)')
    ap.add_argument('--shard', type=int, default=5000)
    ap.add_argument('--min-age', type=float, default=25)
    ap.add_argument('--min-score', type=float, default=0.6)
    ap.add_argument('--min-face', type=int, default=96, help='smallest face box side in pixels')
    args = ap.parse_args()

    det = Detector(os.path.join(args.models, 'det_10g.onnx'))
    rec = Recognizer(os.path.join(args.models, 'w600k_r50.onnx'))
    age = AgeGender(os.path.join(args.models, 'genderage.onnx'))
    os.makedirs(args.out, exist_ok=True)

    files = sorted(p for d in args.images for p in Path(d).rglob('*') if p.suffix.lower() in EXTS)
    print(f'{len(files)} images found', flush=True)
    crops, embs = [], []
    kept = skipped_young = skipped_other = shard = 0
    started = time.time()

    def flush():
        nonlocal crops, embs, shard
        if not crops:
            return
        np.savez(os.path.join(args.out, f'shard_{shard:05d}.npz'), crops=np.stack(crops), emb=np.stack(embs).astype(np.float16))
        shard += 1
        crops, embs = [], []

    for i, path in enumerate(files):
        bgr = cv2.imread(str(path))
        if bgr is None:
            skipped_other += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        found = det.detect(rgb)
        if not found:
            # Close-ups (face touching the edges, like SFHQ) detect with a border around them.
            pad = int(max(rgb.shape[:2]) * 0.25)
            rgb = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            found = det.detect(rgb)
        if not found:
            skipped_other += 1
            continue
        score, box, kps = found[0]
        if score < args.min_score or min(box[2] - box[0], box[3] - box[1]) < args.min_face:
            skipped_other += 1
            continue
        if age.age(rgb, box) < args.min_age:
            skipped_young += 1
            continue
        crops.append(align(rgb, kps, 128))
        embs.append(rec.embed(align(rgb, kps, 112)))
        kept += 1
        if len(crops) >= args.shard:
            flush()
        if i % 1000 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f'{i + 1}/{len(files)} read, {kept} kept, {skipped_young} left out as young-looking, '
                  f'{skipped_other} unclear; {rate:.1f} img/s', flush=True)
        if args.limit and kept >= args.limit:
            break
    flush()
    summary = dict(kept=kept, young=skipped_young, unclear=skipped_other, shards=shard)
    print('done', summary, flush=True)
    with open(os.path.join(args.out, 'prepare_summary.txt'), 'w') as f:
        f.write(repr(summary) + '\n')


if __name__ == '__main__':
    main()
