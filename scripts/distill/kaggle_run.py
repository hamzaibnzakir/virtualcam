"""One Kaggle run of the distillation (the whole job takes 2-3 runs of up to 12 hours each).

Kaggle mounts, read-only, under /kaggle/input/:
  - this kit, as a private dataset (any name; found by looking for distill.py);
  - the SFHQ parts 1, 2 (training) and 3 (checking) datasets;
  - the outputs of earlier runs of this job (kernel_sources), from which it picks up the prepared
    faces and the newest checkpoint.
Each run writes to /kaggle/working/ (kept as the run's output):
  status.json        what happened this run, in one place (read this first)
  run.log            everything printed
  data/train, data/eval   prepared faces (first run only)
  run/               training: ckpt_last.pt, ckpt_best.pt, train_log.csv, val.jsonl, samples/
  export/, eval/     the finished model and its report (last run only)

Settings come from environment variables with these defaults: STEPS=60000 BATCH=32 WIDTH=256
TRAIN_FACES=100000 SESSION_HOURS=11.0.
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

START = time.time()
ROOT = os.environ.get('KAGGLE_ROOT', '/kaggle')  # changed only for local tests
INPUT = os.path.join(ROOT, 'input')
W = os.path.join(ROOT, 'working')
TMP = os.environ.get('MODELS_DIR', '/tmp/vcam')
LOCAL_TEST = os.environ.get('LOCAL_TEST') == '1'  # CPU, no pip: for checking this script off Kaggle
STEPS = int(os.environ.get('STEPS', 60000))
BATCH = int(os.environ.get('BATCH', 32))
WIDTH = int(os.environ.get('WIDTH', 256))
TRAIN_FACES = int(os.environ.get('TRAIN_FACES', 100000))
SESSION_HOURS = float(os.environ.get('SESSION_HOURS', 11.0))
VAL_PAIRS = os.environ.get('VAL_PAIRS', '256')
EVAL_PAIRS = os.environ.get('EVAL_PAIRS', '500')

MODELS = {
    'inswapper_128.onnx': ('https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx',
                           'e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af'),
    'det_10g.onnx': ('https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx',
                     '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'),
    'w600k_r50.onnx': ('https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
                       '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'),
    'genderage.onnx': ('https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx',
                       '4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb'),
}

os.makedirs(W, exist_ok=True)
status = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
log_file = open(os.path.join(W, 'run.log'), 'a')


def say(msg):
    line = f'[{time.strftime("%H:%M:%S")} +{(time.time() - START) / 3600:.2f}h] {msg}'
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()


def save_status(**kw):
    status.update(kw)
    status['hours_used'] = round((time.time() - START) / 3600, 2)
    with open(os.path.join(W, 'status.json'), 'w') as f:
        json.dump(status, f, indent=2)


def run(cmd, name):
    say(f'running {name}: {" ".join(cmd)}')
    t = time.time()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
        p.wait()
    log_file.flush()
    status['stages'].append({'stage': name, 'exit': p.returncode, 'minutes': round((time.time() - t) / 60, 1)})
    save_status()
    if p.returncode != 0:
        raise SystemExit(f'{name} failed with exit code {p.returncode}; see run.log')


def find_kit():
    hits = glob.glob(f'{INPUT}/**/distill.py', recursive=True)
    hits = [h for h in hits if os.path.exists(os.path.join(os.path.dirname(h), 'student.py'))]
    if not hits:
        raise SystemExit(f'the distill kit dataset is not attached (no distill.py under {INPUT})')
    return os.path.dirname(hits[0])


def find_sfhq(part):
    # Kaggle mounts datasets at /kaggle/input/<slug> (or deeper, on newer images).
    for depth in ('*', '*/*', '*/*/*'):
        hits = [d for d in glob.glob(f'{INPUT}/{depth}') if os.path.isdir(d) and f'sfhq-part-{part}' in os.path.basename(d).lower()]
        if hits:
            return hits[0]
    return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch_models(folder):
    os.makedirs(folder, exist_ok=True)
    for name, (url, digest) in MODELS.items():
        path = os.path.join(folder, name)
        if not os.path.exists(path) or sha256(path) != digest:
            say(f'downloading {name}')
            urllib.request.urlretrieve(url, path)
        if sha256(path) != digest:
            raise SystemExit(f'{name}: checksum mismatch')
    say('models downloaded and checked (sha256 OK)')


def previous(pattern):
    """Files from earlier runs' outputs, newest first by modification time."""
    return sorted(glob.glob(f'{INPUT}/**/{pattern}', recursive=True), key=os.path.getmtime, reverse=True)


def main():
    os.makedirs(TMP, exist_ok=True)
    say(f'settings: STEPS={STEPS} BATCH={BATCH} WIDTH={WIDTH} TRAIN_FACES={TRAIN_FACES} SESSION_HOURS={SESSION_HOURS}')
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                             capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        gpu = ''
    say(f'GPU: {gpu or "NONE"}')
    save_status(gpu=gpu or None)
    if not gpu and not LOCAL_TEST:
        save_status(result='no GPU: set the accelerator to GPU (T4) for this notebook')
        raise SystemExit('no GPU')

    kit = find_kit()
    if not LOCAL_TEST:
        run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx2torch', 'lpips', 'onnxconverter-common', 'onnxruntime-gpu'], 'pip install')
    import onnxruntime  # noqa: F401  (after install)

    models = TMP
    fetch_models(models)

    # Prepared faces: from an earlier run if there is one, else made now.
    data = {}
    for split in ('train', 'eval'):
        earlier = previous(f'data/{split}/shard_00000.npz')
        if earlier:
            data[split] = os.path.dirname(earlier[0])
            say(f'using prepared {split} faces from an earlier run: {data[split]}')
    if 'train' not in data or 'eval' not in data:
        p1, p2, p3 = find_sfhq(1), find_sfhq(2), find_sfhq(3)
        say(f'SFHQ datasets: part1={p1} part2={p2} part3={p3}')
        if not (p1 and p2 and p3):
            save_status(result='SFHQ parts 1, 2 and 3 must all be attached as datasets')
            raise SystemExit('missing SFHQ dataset')
        if 'train' not in data:
            data['train'] = os.path.join(W, 'data', 'train')
            run([sys.executable, os.path.join(kit, 'prepare_data.py'), '--images', p1, p2, '--models', models,
                 '--out', data['train'], '--limit', str(TRAIN_FACES)], 'prepare training faces')
        if 'eval' not in data:
            data['eval'] = os.path.join(W, 'data', 'eval')
            run([sys.executable, os.path.join(kit, 'prepare_data.py'), '--images', p3, '--models', models,
                 '--out', data['eval'], '--limit', '2000'], 'prepare evaluation faces')
    for split in ('train', 'eval'):
        summary = os.path.join(data[split], 'prepare_summary.txt')
        if os.path.exists(summary):
            status[f'{split}_faces'] = open(summary).read().strip()

    # Training: continue from the newest checkpoint of an earlier run.
    out = os.path.join(W, 'run')
    os.makedirs(out, exist_ok=True)
    ckpts = previous('run/ckpt_last.pt')
    if ckpts:
        for name in ('ckpt_last.pt', 'ckpt_best.pt', 'train_log.csv', 'val.jsonl'):
            src = os.path.join(os.path.dirname(ckpts[0]), name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, name))
        say(f'continuing from {ckpts[0]}')
    left = SESSION_HOURS - (time.time() - START) / 3600 - 0.6  # leave time to export and evaluate
    if left > 0.2:
        run([sys.executable, os.path.join(kit, 'distill.py'), '--data', data['train'], '--models', models, '--out', out,
             '--width', str(WIDTH), '--batch', str(BATCH), '--steps', str(STEPS), '--hours', f'{left:.2f}',
             '--val-pairs', VAL_PAIRS], 'train')
    vals = [json.loads(l) for l in open(os.path.join(out, 'val.jsonl'))] if os.path.exists(os.path.join(out, 'val.jsonl')) else []
    step = vals[-1]['step'] if vals else 0
    save_status(step=step, steps_target=STEPS, last_val=vals[-1] if vals else None)

    # Export and evaluate whatever is best so far (so every run leaves a usable model).
    if os.path.exists(os.path.join(out, 'ckpt_best.pt')):
        run([sys.executable, os.path.join(kit, 'export.py'), '--ckpt', os.path.join(out, 'ckpt_best.pt'),
             '--models', models, '--out', os.path.join(W, 'export')], 'export')
        students = sorted(glob.glob(os.path.join(W, 'export', '*.onnx')))
        run([sys.executable, os.path.join(kit, 'evaluate.py'), '--data', data['eval'], '--models', models,
             '--students', *students, '--out', os.path.join(W, 'eval'), '--pairs', EVAL_PAIRS], 'evaluate')
        report = json.load(open(os.path.join(W, 'eval', 'report.json')))
        fp16 = next((v for k, v in report['models'].items() if k.endswith('_fp16.onnx')), None)
        targets = None
        if fp16:
            targets = {
                'identity_ratio_vs_teacher >= 0.90': fp16['identity_ratio_vs_teacher'] >= 0.90,
                'match_rate_0.4 >= 0.97': fp16['match_rate_0.4'] >= 0.97,
                'psnr_vs_teacher >= 27': fp16['psnr_vs_teacher'] >= 27,
                'speedup_gpu >= 8': fp16['speedup_gpu'] >= 8,
            }
        save_status(report=report, targets=targets)
    done = step >= STEPS
    save_status(result='training finished' if done else f'stopped at step {step} of {STEPS}: run again to continue',
                finished=done)
    say(status['result'])


if __name__ == '__main__':
    try:
        main()
    except SystemExit as err:
        save_status(error=str(err))
        say(f'STOPPED: {err}')
        raise
    except Exception as err:  # keep a record even on a crash
        import traceback

        save_status(error=traceback.format_exc()[-2000:])
        say(f'CRASHED: {err}')
        raise
