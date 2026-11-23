# Voice models for Brainbox VirtualCam

Two kinds of AI voice, both run live inside the app (src/main/voice):

| | Trained voices (presets) | My voices (uploads) |
|---|---|---|
| Method | RVC v2 (Applio), trained ahead of time | kNN-VC: no training |
| Made from | Kokoro computer speech reading public-domain books | 1-5 min of a voice the user uploads or records (consent check in the app) |
| Ready | after a Kaggle run (`kaggle_voices.py`) | ~1 min on the user's PC |
| App engine | `converter.js` (ContentVec + RMVPE + voice) | `match.js` (WavLM layer 6 + 4-nearest match + HiFi-GAN) |
| Quality | most natural | close to the uploaded voice (0.87 vs 0.96 same-speaker similarity in tests) |

## Files
- `kaggle_voices.py`: the whole job on a free Kaggle T4 (see `HERMES_PROMPT_VOICES.md`).
- `export_shared.py`: contentvec.onnx + rmvpe.onnx. `export_clone.py`: wavlm6.onnx, knn-vocoder.onnx, knn-match.onnx.
- `half_storage.py`: float16 weights, float32 maths (half the download, runs on any CPU or GPU).
- `convert_file.py`: whole-file conversion with a trained voice (the reference the app is checked against).
- `make_test_models.py`: tiny stand-in models in `testdata/voice` for the app's self-test.

## Checks done here (CPU)
- App engine vs reference (trained-voice path, RVC base model): mel correlation 0.95, pitch moved
  102 Hz → 203 Hz as asked, no clicks between blocks.
- My voices: prepared from 3 min of a LibriSpeech speaker; speaker similarity of the converted speech
  0.55 → 0.87 live (0.88 whole-file); preparing took 28 s on 2 CPU cores.
- `node scripts/test-voice.cjs` and `node scripts/test-my-voice.cjs` stream a recording through the
  engines in 20 ms chunks, like the microphone does.
