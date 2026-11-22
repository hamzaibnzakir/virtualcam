"""Compares the exported student(s) with the teacher on faces never used in training, with ONNX Runtime
(the same runtime the app uses). Writes <out>/report.json and <out>/compare.jpg
(rows: camera face | teacher | each student).

  python evaluate.py --data /data/eval_prepared --models /models \
      --students /runs/fast1/export/inswapper_fast_256.onnx /runs/fast1/export/inswapper_fast_256_fp16.onnx \
      --out /runs/fast1/eval
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from faces import emap_from_swapper, latent_from_embedding  # noqa: E402


def session(path, provider):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, opts, providers=[provider])


def speed(path, provider, runs=50):
    s = session(path, provider)
    feeds = {'target': np.random.rand(1, 3, 128, 128).astype(np.float32), 'source': np.random.rand(1, 512).astype(np.float32)}
    for _ in range(5):
        s.run(None, feeds)
    t = time.perf_counter()
    for _ in range(runs):
        s.run(None, feeds)
    return (time.perf_counter() - t) / runs * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='prepare_data.py output of held-out faces (e.g. SFHQ part 3)')
    ap.add_argument('--models', required=True)
    ap.add_argument('--students', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pairs', type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    crops, embs = [], []
    for f in sorted(glob.glob(os.path.join(args.data, 'shard_*.npz'))):
        d = np.load(f)
        crops.append(d['crops'])
        embs.append(d['emb'].astype(np.float32))
    crops, embs = np.concatenate(crops), np.concatenate(embs)
    rng = np.random.default_rng(7)
    n = min(args.pairs, len(crops))
    t_idx = rng.integers(0, len(crops), n)
    s_idx = (t_idx + rng.integers(1, len(crops), n)) % len(crops)  # always someone else
    teacher_path = os.path.join(args.models, 'inswapper_128.onnx')
    emap = emap_from_swapper(teacher_path)
    gpu = 'CUDAExecutionProvider' if 'CUDAExecutionProvider' in ort.get_available_providers() else 'CPUExecutionProvider'
    rec = session(os.path.join(args.models, 'w600k_r50.onnx'), gpu)
    rec_in = rec.get_inputs()[0].name

    def identity(out128):
        face = out128[:, :, 0:112, 8:120] * 2 - 1
        e = rec.run(None, {rec_in: face.astype(np.float32)})[0][0]
        return e / np.linalg.norm(e)

    models = {'teacher': teacher_path, **{os.path.basename(p): p for p in args.students}}
    sessions = {k: session(p, gpu) for k, p in models.items()}
    outs = {k: [] for k in models}
    ids = {k: [] for k in models}
    for i in range(n):
        target = (crops[t_idx[i]].astype(np.float32) / 255).transpose(2, 0, 1)[None]
        latent = latent_from_embedding(embs[s_idx[i]], emap)[None].astype(np.float32)
        for k, s in sessions.items():
            o = s.run(None, {'target': target, 'source': latent})[0]
            ids[k].append(float(identity(o) @ embs[s_idx[i]]))
            if i < 8 or k != 'teacher':
                outs[k].append(o if i < 8 else None)
            if k == 'teacher':
                teacher_out = o
            else:
                outs.setdefault(k + ':psnr', []).append(float(10 * np.log10(1 / max(1e-10, np.mean((o - teacher_out) ** 2)))))
                outs.setdefault(k + ':l1', []).append(float(np.mean(np.abs(o - teacher_out))))
                outs.setdefault(k + ':idvt', []).append(float(identity(o) @ identity(teacher_out)))

    report = {'pairs': n, 'gpu_provider': gpu, 'models': {}}
    for k, p in models.items():
        r = {
            'file': os.path.basename(p),
            'mb': round(os.path.getsize(p) / 1e6, 1),
            'identity_to_source': round(float(np.mean(ids[k])), 4),
            'match_rate_0.4': round(float(np.mean(np.array(ids[k]) >= 0.4)), 3),
            'ms_gpu': round(speed(p, gpu), 2),
            'ms_cpu': round(speed(p, 'CPUExecutionProvider', runs=10), 1),
        }
        if k != 'teacher':
            r['psnr_vs_teacher'] = round(float(np.mean(outs[k + ':psnr'])), 2)
            r['l1_vs_teacher'] = round(float(np.mean(outs[k + ':l1'])), 4)
            r['identity_vs_teacher_output'] = round(float(np.mean(outs[k + ':idvt'])), 4)
            r['identity_ratio_vs_teacher'] = round(r['identity_to_source'] / max(1e-6, float(np.mean(ids['teacher']))), 3)
            r['speedup_gpu'] = None
        report['models'][k] = r
    for k, r in report['models'].items():
        if k != 'teacher':
            r['speedup_gpu'] = round(report['models']['teacher']['ms_gpu'] / r['ms_gpu'], 2)
            r['speedup_cpu'] = round(report['models']['teacher']['ms_cpu'] / r['ms_cpu'], 2)
    with open(os.path.join(args.out, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    rows = []
    for i in range(min(8, n)):
        tile = [crops[t_idx[i]]] + [(outs[k][i][0].transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8) for k in models]
        rows.append(np.concatenate(tile, 1))
    cv2.imwrite(os.path.join(args.out, 'compare.jpg'), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()
