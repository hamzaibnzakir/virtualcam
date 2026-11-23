# Task for Hermes: make and publish Brainbox VirtualCam's AI voices (Kaggle, free GPU)

Hamza wants this done end to end without his help. One Kaggle run (about 5 to 7 hours on the free
T4 GPU) makes everything; then you publish the files so the app can download them. The code is in
the repo, in `scripts/voice-train/`. The method is fixed and tested: don't change it.

## What the run makes
1. **"My voices" models** (no training, done in the first ~20 minutes): let users upload or record a
   voice and use it a minute later. `wavlm6.onnx` (~178 MB), `knn-vocoder.onnx` (~33 MB), `knn-match.onnx` (tiny).
2. **Four trained voices**: Young man, Young woman, Older man, Older woman (`voice-<id>.onnx`, ~55 MB
   each), trained only on computer-made speech (Kokoro text-to-speech reading public-domain books).
   No real person's voice is used.
3. **Two shared speech models** the trained voices need: `contentvec.onnx` (~189 MB), `rmvpe.onnx` (~181 MB).
4. `voice-models.json` and `my-voice-models.json`: the manifests the app reads. They already point at
   the release URLs below.

All of it is open licensed (Kokoro Apache-2.0; RVC/Applio, ContentVec, RMVPE, WavLM, kNN-VC: MIT),
so publishing it in the public releases repo is fine.

## Links
- Repo (private): https://github.com/hamzaibnzakir/virtualcam → `scripts/voice-train/` (`kaggle_voices.py` is the run script)
- Kaggle metadata templates: `scripts/voice-train/kaggle/`
- Releases repo (public, made by the earlier task `docs/HERMES_PROMPT_UPDATES.md`): https://github.com/hamzaibnzakir/brainbox-virtualcam-releases
- Kaggle API: https://github.com/Kaggle/kaggle-api

## Rules
1. **Cost $0**: Kaggle's free tier only. One run of up to 12 h; check the GPU quota first (kaggle.com → Settings).
   If the face model distillation is still using the quota, wait for it to finish, or for the weekly reset.
2. Kaggle dataset and kernel stay **private**. Only the files in `upload/` get published, in the releases repo.
3. Never add real people's recordings. Don't change the Kokoro voices, the books or the training settings,
   except as rule 4 says.
4. If something fails, read `run.log` and `status.json`, fix only what's needed (a package version, a
   path), note it, and run again. To redo only some voices, make run 2 the same way as run 1 (`RUN` = 2) and:
   add `"kernel_sources": ["<username>/brainbox-voices-1"]` to its kernel-metadata.json (the script keeps
   everything run 1 made), and put `os.environ.setdefault("VOICES", "older-man,older-woman")` (the
   missing ones) right after the imports at the top of run 2's `kaggle_voices.py`.
5. Notes short and factual: commands, URLs, results.

## Steps
**A. Kit dataset**
```bash
git clone https://github.com/hamzaibnzakir/virtualcam.git   # or git pull
mkdir -p ~/bbx-voices/kit && cp virtualcam/scripts/voice-train/*.py ~/bbx-voices/kit/
sed "s/KAGGLE_USERNAME/<username>/" virtualcam/scripts/voice-train/kaggle/dataset-metadata.template.json > ~/bbx-voices/kit/dataset-metadata.json
kaggle datasets create -p ~/bbx-voices/kit      # later updates: kaggle datasets version -p ~/bbx-voices/kit -m "msg"
```

**B. The run**
```bash
mkdir -p ~/bbx-voices/run1 && cp virtualcam/scripts/voice-train/kaggle_voices.py ~/bbx-voices/run1/
sed "s/KAGGLE_USERNAME/<username>/g; s/RUN/1/g" virtualcam/scripts/voice-train/kaggle/kernel-metadata.template.json > ~/bbx-voices/run1/kernel-metadata.json
kaggle kernels push -p ~/bbx-voices/run1
kaggle kernels status <username>/brainbox-voices-1     # every ~30 min
```
- Within 15 minutes, check the log shows the GPU: `status.json` → stage `setup` → `"gpu": "Tesla T4"`
  (visible on the kernel page's log). If it's `null`, set the accelerator to "GPU T4" on the kernel's
  web settings and run again.
- When it's finished: `kaggle kernels output <username>/brainbox-voices-1 -p ~/bbx-voices/out1`

**C. Check before publishing**
- `status.json`: stages `my voice models`, `shared models`, the four `voice <id>` stages, `samples` and `manifest` all `"ok": true`.
- Listen to `samples/*.wav`: each `<voice>-from-am_echo.wav` must be clear speech in that voice (young/older,
  man/woman), with no robotic buzz or long dropouts. Note anything odd in the report.
- `upload/` has: `voice-models.json`, `my-voice-models.json`, `contentvec.onnx`, `rmvpe.onnx`,
  `wavlm6.onnx`, `knn-vocoder.onnx`, `knn-match.onnx`, and four `voice-*.onnx`.
- Check each file's sha256 matches its entry in the two manifests (`sha256sum upload/*.onnx`).

**D. Publish** (needs the releases repo from `docs/HERMES_PROMPT_UPDATES.md`; do that task first if it isn't done)
```bash
cd ~/bbx-voices/out1/upload
gh release create models-voice --repo hamzaibnzakir/brainbox-virtualcam-releases \
  --title "Voice models" --notes "AI voices and My voices models for Brainbox VirtualCam (downloaded by the app)." \
  *.onnx voice-models.json my-voice-models.json
```
If "My voices" models are ready but some trained voices failed, publish what's ready anyway (the
manifests only list what was made), then redo the rest and upload with `gh release upload models-voice --clobber ...`.

Verify the app's links work (each must download, not 404):
```bash
curl -sIL https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/download/models-voice/voice-models.json | grep -i "^HTTP"
curl -sIL https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/download/models-voice/my-voice-models.json | grep -i "^HTTP"
```

## Report
```
Kernel URL / run time (status.json → hours):
GPU line:
Stages (status.json → stages: name, ok, hours):
Voices (status.json → voices: speech_minutes, pitch_hz, train_hours, onnx_mb):
Samples: how each voice sounds (one line each), attach 2 samples
Release URL + each file's name, size, sha256:
curl checks (both manifests: HTTP 200 after redirects?):
Problems and fixes:
GPU quota left:
```
