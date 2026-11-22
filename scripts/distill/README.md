# Fast swap model (distillation kit)

Trains a small copy ("student") of the face swap model inswapper_128 ("teacher"). The student
learns to produce the teacher's output for any face and identity, at about 1/17 of the computing
work: 21 instead of 350 billion operations per face. Its inputs and output are the same as the
teacher's, so the VirtualCam app can load it in its place.

| | Teacher (inswapper_128) | Student, width 256 (default) | Student, width 384 |
|---|---|---|---|
| Compute per face | ~350 GFLOPs | 20.7 GFLOPs | 40.2 GFLOPs |
| File (fp16) | 278 MB | ~23 MB | ~43 MB |
| Measured on a 2-core test machine (CPU) | 2,340 ms | 153 ms | — |
| Expected on a Quadro T2000 (DirectML) | 232 ms (user's log) | ~15–30 ms | ~30–50 ms |

**Licence.** InsightFace released inswapper for non-commercial research only. A student trained to
copy it should be treated the same way: personal use, not for sale.

## What runs where

| File | Job |
|---|---|
| `setup_vast.sh` | Installs packages, downloads and checks the 4 models (sha256), downloads SFHQ parts 1–3 from Kaggle |
| `prepare_data.py` | Photos → aligned 128×128 face crops + identity embeddings, in `.npz` shards. Leaves out faces that look under 25 |
| `distill.py` | Training: teacher and student on the same inputs; L1 + LPIPS loss towards the teacher's picture, plus an identity loss. Logs, samples and checkpoints; resumes after an interruption |
| `export.py` | Best checkpoint → `inswapper_fast_<w>.onnx` and `_fp16.onnx`, with the same inputs/outputs as the teacher and its identity matrix embedded |
| `evaluate.py` | Teacher vs student(s) on 500 held-out pairs (SFHQ part 3): identity, picture difference, speed on the GPU and CPU. Writes `report.json` + `compare.jpg` |
| `faces.py`, `student.py` | Shared code: face finding/alignment like the app; the student network |

All of it was run end to end on a CPU with a handful of faces before release. Only the full-size
GPU run is new.

## Free route: Kaggle

`kaggle_run.py` runs the whole job on Kaggle's free T4 GPU in 2–3 runs of up to 12 hours each. Each
run picks up the prepared faces and the newest checkpoint from the runs before it, and leaves an
exported, evaluated model plus a `status.json`. It uses float16 on the T4, after a start-up check
that the teacher gives the same pictures in float16. Templates are in `kaggle/`; the step-by-step
instructions are in `HERMES_PROMPT_KAGGLE.md`. Both runs were simulated end to end on a CPU:
preparation, training, export, evaluation, and the second run resuming the first.

## Paid route: vast.ai

### 1. Rent the machine (vast.ai)

- GPU: one **RTX 4090** (24 GB). An RTX 3090 also works, ~1.7× slower. An A100 or H100 is fine if it's cheap.
- Image: **PyTorch 2.x with CUDA 12.x** (vast's "PyTorch (cuDNN devel)" template).
- Disk: **≥ 200 GB** (the SFHQ download is about 60 GB zipped, plus unzipped copies and shards).
- RAM: ≥ 32 GB. Upload/download speed: ≥ 500 Mbps if you can.
- Pick a host with reliability ≥ 98%. Use on-demand, not interruptible, unless you watch it.
- Expected cost: 6–10 GPU-hours plus 1–2 hours of setup and data. About $5–10 on a 4090.

### 2. Set up

```bash
cd /workspace && git clone https://github.com/hamzaibnzakir/virtualcam.git   # private: use a token with read access
cd virtualcam/scripts/distill
mkdir -p ~/.kaggle && cp /path/to/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
bash setup_vast.sh 2>&1 | tee /workspace/setup.log
```

### 3. Prepare the faces (≈1–2 h, use `tmux`)

```bash
python prepare_data.py --images /workspace/data/sfhq1 /workspace/data/sfhq2 \
  --models /workspace/models --out /workspace/data/train --limit 120000 2>&1 | tee /workspace/prepare_train.log
python prepare_data.py --images /workspace/data/sfhq3 \
  --models /workspace/models --out /workspace/data/eval --limit 2000 2>&1 | tee /workspace/prepare_eval.log
```

### 4. Pilot run (≈10 min): check speed and that the loss falls

```bash
python distill.py --data /workspace/data/train --models /workspace/models --out /workspace/runs/pilot \
  --steps 2000 --val-every 500 2>&1 | tee /workspace/pilot.log
```

From the log, `it/s` × 3600 = steps per hour. Pick `--steps` so that the full run fits about 8 hours
(default 80,000). Look at `runs/pilot/samples/step_002000.jpg` (columns: camera face | teacher | student).
The student should already be a blurry version of the teacher.

### 5. Full run (6–10 h)

```bash
python distill.py --data /workspace/data/train --models /workspace/models --out /workspace/runs/fast256 \
  --width 256 --steps 80000 --hours 10 2>&1 | tee /workspace/runs/fast256.log
```

If it stops, run the same command again: it resumes from `ckpt_last.pt`.

### 6. Export and evaluate

```bash
python export.py --ckpt /workspace/runs/fast256/ckpt_best.pt --models /workspace/models --out /workspace/runs/fast256/export
python evaluate.py --data /workspace/data/eval --models /workspace/models \
  --students /workspace/runs/fast256/export/inswapper_fast_256.onnx /workspace/runs/fast256/export/inswapper_fast_256_fp16.onnx \
  --out /workspace/runs/fast256/eval
```

### When is it good enough?

From `report.json`, for `inswapper_fast_256_fp16.onnx`:

| Check | Target | Meaning |
|---|---|---|
| `identity_ratio_vs_teacher` | **≥ 0.90** | keeps at least 90% of the teacher's likeness to the photo |
| `match_rate_0.4` | ≥ 0.97 | almost every swap is recognisably the photo's person |
| `psnr_vs_teacher` | ≥ 27 dB | the picture is close to the teacher's |
| `speedup_gpu` | ≥ 8× | |

If the identity ratio ends below 0.90, run once more with `--width 384`. It's slower (~40 GFLOPs),
but it's still ~9× faster than the teacher. Compare the two in one `evaluate.py` call.

## What comes back

Put these in one folder and upload it (see the prompt): `export/*.onnx`, `eval/report.json`,
`eval/compare.jpg`, `train_log.csv`, `val.jsonl`, the last 3 `samples/*.jpg`, `args.json`, the
prepare/pilot/training logs, and `ckpt_best.pt` (to train further later).
