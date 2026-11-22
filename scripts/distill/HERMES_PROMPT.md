# Task for Hermes: train VirtualCam's fast face swap model on vast.ai

You're training a small, fast copy of the face swap model that VirtualCam's Local AI uses, on a rented
GPU. The full guide and all scripts are in the repo at `scripts/distill/`. Read
`scripts/distill/README.md` before you start, and follow it step by step. These instructions take
priority if the two disagree.

## Goal
Produce `inswapper_fast_256_fp16.onnx`: a drop-in replacement for inswapper_128 (same inputs `target`
1×3×128×128 and `source` 1×512, output `output` 1×3×128×128). It should be about 8–15× faster and keep
≥ 90% of the teacher's identity score on held-out faces.

## Links
- Repo (private): https://github.com/hamzaibnzakir/virtualcam (folder `scripts/distill/`)
- Teacher model: https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
  (sha256 `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af`, 554 MB)
- Helper models (insightface buffalo_l): https://huggingface.co/public-data/insightface/tree/main/models/buffalo_l
  (det_10g, w600k_r50, genderage; `setup_vast.sh` downloads and checks them)
- Training faces, AI-generated with no real people (SFHQ):
  - https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-1
  - https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-2
  - https://www.kaggle.com/datasets/selfishgene/synthetic-faces-high-quality-sfhq-part-3 (evaluation only, never training)
- vast.ai console: https://cloud.vast.ai

## What you need from Hamza before starting
- A vast.ai account with credit (or its API key). **Budget cap: $25.**
- A Kaggle API token (`kaggle.json`).
- A GitHub token with read access to the repo, and write access if you upload the results as a GitHub release.

## Rules
1. **Spending:** don't exceed $25 in total. Check the running cost at every report. If the pilot
   projects the full run above $20, stop and ask before continuing.
2. **Destroy the instance** (not just stop it) once the results are uploaded and verified, so storage
   isn't billed. Write down the instance ID and the final cost.
3. **Data:** use only the SFHQ datasets above. Don't add real people's photos. Keep `prepare_data.py`'s
   age filter on (`--min-age 25`). Never train on SFHQ part 3.
4. **Licence:** the teacher is licensed for non-commercial research only, so the trained model is too.
   Keep it in a private place, and don't publish it publicly.
5. Run long steps inside `tmux` and tee every log to `/workspace/*.log`, so a dropped SSH session doesn't kill them.
6. If a step fails, read its log, fix only what's needed (a package version, a path), note the fix,
   and continue. Don't change the training method or the loss weights without asking. Exception: if
   the pilot runs out of GPU memory, halve `--batch`.
7. Keep notes short and factual: decisions, URLs, paths, commands, verified results.

## Steps (details in the README)
1. Rent: 1× RTX 4090 (or 3090/A100), PyTorch + CUDA 12 image, ≥ 200 GB disk, ≥ 32 GB RAM, reliability ≥ 98%, on-demand.
2. Clone the repo, add `kaggle.json`, then run `bash setup_vast.sh`. All 4 sha256 checks must print OK.
3. Prepare 120,000 training faces (parts 1+2) and 2,000 evaluation faces (part 3).
4. Run the 2,000-step pilot, then **send report #1**.
5. Run the full training (default 80,000 steps; `--hours 10` cap). Send a short progress note every
   ~2 hours: step, it/s, the latest `VAL` line, cost so far.
6. Export, evaluate, check against the targets below.
7. If `identity_ratio_vs_teacher` < 0.90: one extra run with `--width 384`, but only if the budget
   allows. Evaluate both together.
8. Upload the results (below), verify the download works, destroy the instance, and **send the final report**.

## Targets (from `eval/report.json`, fp16 student)
- `identity_ratio_vs_teacher` ≥ 0.90
- `match_rate_0.4` ≥ 0.97
- `psnr_vs_teacher` ≥ 27
- `speedup_gpu` ≥ 8

## Where to upload
Make a **private** GitHub release on the repo:

```bash
gh release create distill-v1 --repo hamzaibnzakir/virtualcam --title "Fast swap model v1" --notes-file final_report.md <files>
```

Release assets are limited to 2 GB each. If that's not possible, use a private Hugging Face model
repo under Hamza's account. Files to upload:
- `inswapper_fast_256.onnx`, `inswapper_fast_256_fp16.onnx` (and the `_384` ones if trained)
- `report.json`, `compare.jpg`
- `train_log.csv`, `val.jsonl`, `args.json`
- the last 3 sample images
- `ckpt_best.pt`
- all `*.log` files

## Report #1 (after the pilot)
```
GPU / host / $ per hour:
Setup: sha256 checks OK? package versions (torch, onnxruntime-gpu, CUDA):
Faces prepared: train kept / young left out / unclear; eval kept   (from prepare_summary.txt)
Pilot: it/s, loss at step 100 vs 2000, last VAL line (psnr, id_ratio)
Projected full run: steps, hours, cost
Sample image: runs/pilot/samples/step_002000.jpg (attach)
Problems and fixes so far:
Go / no-go for the full run:
```

## Final report
```
Result: targets met? (yes/no per target)
report.json (paste it in full)
compare.jpg (attach): columns are camera face | teacher | student(s)
Training: steps done, hours, final VAL line, best checkpoint step
Files uploaded: release/repo URL + each file's name, size and sha256
Cost: total $, GPU hours, instance ID, destroyed at (time)
Problems and fixes:
What you'd change next time:
```
