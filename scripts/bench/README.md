# Lite engine benchmark

Measures how much the Lite face warp makes the camera face look like the photo, on AI-generated
adult faces (SFHQ sample; faces that could read as under 18 are excluded after a manual check).

```
export BENCH_FACES=…  BENCH_MODELS=…  BENCH_CACHE=…   # optional shared folders (downloaded if missing)
node scripts/bench/bench-lite.mjs --split train   --engine lite            # tune on this
node scripts/bench/bench-lite.mjs --split holdout --engine lite            # report on this
node scripts/bench/bench-lite.mjs --split holdout --engine local           # the Local AI engine (the "100%" mark)
node scripts/bench/bench-lite.mjs --split holdout --engine original        # no change (the floor)
node scripts/bench/bench-lite.mjs --params '{"shading":0.3}' --limit 20 --images 6
```

- `idRef`: ArcFace cosine between the output face and the photo. Higher = more like the photo.
- `idOrig`: the same against the original camera face. Lower = more changed.
- `pose`: average movement of the 5 face points (eyes, nose, mouth corners) from the camera's, as a
  share of the eye distance. It must stay small, so the result still follows the camera's head and
  expression instead of pasting the photo.
- `matchRate`: share of pairs with `idRef` ≥ 0.4 (roughly "recognised as the photo's person").
- Pairs: 66 train / 36 holdout, different people in each. Tune on train only; holdout is for claims.

The renderer is the real `src/renderer/engine/ai-lite.js` (MediaPipe tracking + WebGL), run in a hidden
Electron window for a few frames per pair. Scoring uses the InsightFace models the app's Local engine uses.
