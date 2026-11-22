#!/usr/bin/env python3
"""Side-by-side proof sheet from several finished benchmark runs (test tool only).

Usage:
  python3 scripts/bench/proof-sheet.py OUT.jpg FACES_DIR CONFIG_RUN \
      "Label=RUN_DIR" "Label=RUN_DIR" ... [--rows i,j,k] [--tile 220]

CONFIG_RUN is any run of the same split that wrote the original camera frames
(<target>.orig.rgba). Each RUN_DIR must hold <target>.out.rgba and results.json.
Columns: photo | camera | one column per run, with that run's identity score.
"""
import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720


def load_font(size):
    for name in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def main():
    args = [a for a in sys.argv[1:]]
    rows_arg, tile = None, 220
    if '--rows' in args:
        i = args.index('--rows')
        rows_arg = [int(v) for v in args[i + 1].split(',')]
        del args[i:i + 2]
    if '--tile' in args:
        i = args.index('--tile')
        tile = int(args[i + 1])
        del args[i:i + 2]
    out, faces, config_run = args[0], args[1], args[2]
    runs = [a.split('=', 1) for a in args[3:]]

    pairs = json.load(open(os.path.join(config_run, 'config.json')))['pairs']
    scores = []
    for _, d in runs:
        res = json.load(open(os.path.join(d, 'results.json')))
        scores.append({r['id']: r.get('idRef') for r in res['rows']})
    picked = rows_arg if rows_arg is not None else list(range(len(pairs)))

    head = 34
    cols = 2 + len(runs)
    sheet = Image.new('RGB', (tile * cols, head + tile * len(picked)), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    f_head, f_score = load_font(15), load_font(16)
    for c, label in enumerate(['Photo', 'Camera'] + [r[0] for r in runs]):
        draw.text((c * tile + 8, 9), label, fill=(235, 235, 235), font=f_head)

    for row, idx in enumerate(picked):
        p = pairs[idx]
        f = p['frame']
        pad = 20
        box = (f['x'] - pad, f['y'] - pad, f['x'] + f['size'] + pad, f['y'] + f['size'] + pad)
        y = head + row * tile
        ref = Image.open(os.path.join(faces, p['ref'])).convert('RGB').resize((tile, tile))
        sheet.paste(ref, (0, y))
        orig_file = os.path.join(config_run, p['target'] + '.orig.rgba')
        orig = Image.frombytes('RGBA', (W, H), open(orig_file, 'rb').read()).convert('RGB')
        sheet.paste(orig.crop(box).resize((tile, tile)), (tile, y))
        pid = p['split'] + ':' + p['target']
        for c, (_, d) in enumerate(runs):
            x = (2 + c) * tile
            path = os.path.join(d, p['target'] + '.out.rgba')
            img = Image.frombytes('RGBA', (W, H), open(path, 'rb').read()).convert('RGB')
            sheet.paste(img.crop(box).resize((tile, tile)), (x, y))
            s = scores[c].get(pid)
            text = 'match %.2f' % s if s is not None else 'no face'
            ok = s is not None and s >= 0.4
            draw.rectangle([x, y, x + 104, y + 22], fill=(0, 0, 0))
            draw.text((x + 5, y + 3), text, fill=(120, 230, 140) if ok else (250, 190, 90), font=f_score)
    sheet.save(out, quality=90)
    print(out)


if __name__ == '__main__':
    main()
