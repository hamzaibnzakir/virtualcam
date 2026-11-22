# Task for Hermes: train VirtualCam's fast face swap model on Kaggle (free GPU)

Hamza wants this done end to end without his help, except for the one-time account steps marked
**[Hamza]**. You'll train a small, fast copy of the face swap model that VirtualCam's Local AI uses,
on Kaggle's free T4 GPU, across 2 (maybe 3) runs of up to 12 hours each. Everything runs from the
command line with the Kaggle API. The code is in the repo, in `scripts/distill/`.

The training method, the student network and the checks are fixed and tested. Your job is to set up
the runs, chain them, watch them, and report. Don't change the method or the loss weights.

## Goal
`inswapper_fast_256_fp16.onnx`: a drop-in replacement for inswapper_128 (inputs `target` 1×3×128×128
and `source` 1×512, output `output` 1×3×128×128). About 10× less work per face, keeping ≥ 90% of the
teacher's identity score on faces never used in training.

## Links
- Repo (private): https://github.com/hamzaibnzakir/virtualcam → `scripts/distill/` (read `README.md` there too)
- The Kaggle run script: `scripts/distill/kaggle_run.py`. Metadata templates: `scripts/distill/kaggle/`
- Teacher model, downloaded by the script with a sha256 check:
  https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx
- Helper models (insightface buffalo_l, also downloaded by the script): https://huggingface.co/public-data/insightface/tree/main/models/buffalo_l
- Training faces, AI-generated with no real people (Kaggle datasets, attached as inputs, nothing to download):
  - `selfishgene/synthetic-faces-high-quality-sfhq-part-1` and `selfishgene/synthetic-faces-high-quality-sfhq-part-2`: training
  - `selfishgene/synthetic-faces-high-quality-sfhq-part-3`: final check only
- Kaggle API docs: https://github.com/Kaggle/kaggle-api

## One-time account steps
1. **[Hamza]** Kaggle account, **phone-verified**: kaggle.com → Settings → Phone verification. Without
   it, Kaggle gives no GPU and no internet.
2. **[Hamza or you, if you can use his logged-in browser]** Kaggle → Settings → API → "Create New
   Token". Save `kaggle.json` to `~/.kaggle/kaggle.json` (Windows: `%USERPROFILE%\.kaggle\kaggle.json`).
3. You: `pip install kaggle`, then check it works with `kaggle datasets list -s sfhq`. Note the Kaggle
   username (`kaggle config view`): it's `KAGGLE_USERNAME` below.

## Rules
1. **Cost: $0.** Kaggle's free tier only. Never enable paid compute or other paid add-ons.
2. Kaggle gives about **30 GPU hours a week**, with runs capped at 12 hours. The plan uses 2 runs of
   ~11 h. Check the remaining quota (kaggle.com → Settings) before each run. If it isn't enough, wait
   for the weekly reset and say so in a report.
3. **Data:** only the SFHQ datasets above. No real people's photos. Keep the age filter as it is. Never train on part 3.
4. Everything stays **private**: the kit dataset, the notebooks/kernels and the outputs. The teacher
   model is licensed for non-commercial research only, so the trained model is too. Never publish it.
5. If something fails: read `run.log` and `status.json` from the run's output, fix only what's needed
   (a path, a package version, a dataset slug), note the fix, and rerun. Exception: if the GPU runs out
   of memory, change `BATCH` in `kaggle_run.py` from 32 to 16 before you push.
6. Keep notes short and factual: decisions, URLs, commands, kernel versions, verified results.

## Steps
**A. Upload the kit as a private dataset** (any machine with the repo):
```bash
git clone https://github.com/hamzaibnzakir/virtualcam.git   # or git pull if you have it
mkdir -p ~/vcam-kaggle/kit && cp virtualcam/scripts/distill/*.py ~/vcam-kaggle/kit/
sed "s/KAGGLE_USERNAME/<username>/" virtualcam/scripts/distill/kaggle/dataset-metadata.template.json > ~/vcam-kaggle/kit/dataset-metadata.json
kaggle datasets create -p ~/vcam-kaggle/kit          # private by default; to update later: kaggle datasets version -p ... -m "msg"
```

**B. Run 1:** prepares the faces (~1–2 h), then trains for the rest of the 11 h. Every run also
exports and evaluates its best model so far, so each run leaves a usable result.
```bash
mkdir -p ~/vcam-kaggle/run1 && cp virtualcam/scripts/distill/kaggle_run.py ~/vcam-kaggle/run1/
sed "s/KAGGLE_USERNAME/<username>/g; s/RUN/1/g" virtualcam/scripts/distill/kaggle/kernel-metadata.template.json > ~/vcam-kaggle/run1/kernel-metadata.json
kaggle kernels push -p ~/vcam-kaggle/run1
kaggle kernels status <username>/virtualcam-distill-1        # check every ~30 min until it says complete
```
- **Within 10 minutes of the push**, check that it's on a GPU. In the kernel's log (on the website,
  or the output once finished), the line `GPU: Tesla T4 ...` must appear. If it says `GPU: NONE`, set
  the accelerator to "GPU T4" in the kernel's web settings (Hamza's logged-in browser) and run again.
- When finished: `kaggle kernels output <username>/virtualcam-distill-1 -p ~/vcam-kaggle/out1`.
  Then read `out1/status.json` and **send report #1**.

**C. Run 2:** continues training from run 1's checkpoint, using run 1's prepared faces:
```bash
mkdir -p ~/vcam-kaggle/run2 && cp virtualcam/scripts/distill/kaggle_run.py ~/vcam-kaggle/run2/
sed "s/KAGGLE_USERNAME/<username>/g; s/RUN/2/g" virtualcam/scripts/distill/kaggle/kernel-metadata.template.json > ~/vcam-kaggle/run2/kernel-metadata.json
# edit run2/kernel-metadata.json: "kernel_sources": ["<username>/virtualcam-distill-1"]
kaggle kernels push -p ~/vcam-kaggle/run2
```
Download its output to `~/vcam-kaggle/out2`.

**D. Run 3, only if** `status.json` says `stopped at step X of 60000` and quota allows. Same as
run 2, with `RUN` = 3 and `"kernel_sources": ["<username>/virtualcam-distill-1", "<username>/virtualcam-distill-2"]`.

**E. Deliver:** when `status.json` says `training finished`, put these in one folder:
- `export/*.onnx`
- `eval/report.json` and `eval/compare.jpg`
- `run/train_log.csv`, `run/val.jsonl`, the last 3 `run/samples/*.jpg`
- `run/ckpt_best.pt`
- `status.json` and `run.log` from every run

Create a **private** GitHub release:
`gh release create distill-v1 --repo hamzaibnzakir/virtualcam --title "Fast swap model v1" --notes-file final_report.md <files>`.
If you can't, put the folder in Hamza's Google Drive. Verify the download works. Then **send the final report**.

## Targets
From `status.json` → `targets` (computed from `eval/report.json` for the fp16 student):
`identity_ratio_vs_teacher >= 0.90`, `match_rate_0.4 >= 0.97`, `psnr_vs_teacher >= 27`, `speedup_gpu >= 8`.
If only the identity target is missed at the end, report it. Don't start extra runs: Claude will decide
whether to train the wider model (`WIDTH=384`).

## Report #1 (after run 1)
```
Kaggle username / kit dataset URL / kernel URL:
GPU line from run.log:
Stages and minutes (status.json → stages):
Faces prepared: train_faces / eval_faces (status.json)
Training: step reached of 60000, steps per hour (train_log.csv: it_per_s x 3600), last VAL line
Targets so far (status.json → targets):
GPU quota left this week:
Problems and fixes:
Plan: run 2 now / wait for quota reset (date)
```

## Final report
```
Result: targets met? (each yes/no)
status.json → report (paste in full)
compare.jpg (attach): columns are camera face | teacher | student fp32 | student fp16
Training: total steps, runs used, GPU hours, best checkpoint step
Files: release URL (or Drive link) + each file's name, size and sha256
Problems and fixes:
What you'd change next time:
```
