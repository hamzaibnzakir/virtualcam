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

## Motion mode (`--motion`, Lite only, off by default)

A live camera moves; the still pairs don't. With `--motion` each pair is fed through `lite.tick` as a
moving camera: 8 still frames to settle, then 30 frames (`--motion-frames`) where the head moves
sideways and back, rolls +8° to -8°, scales 0.9 to 1.1, narrows to 85% width and back (the
foreshortening of a ~30° head turn, so the face changes shape and not only position) and the exposure
dims to 75%, easing in and out, then holds still; then 2 black frames (`--motion-dropout`, a camera hiccup) after which the face
returns slightly to the side. Every frame gets seeded sensor noise (`--motion-noise`, in 0-255 levels)
and a simulated 30 fps timestamp, so the smoothing sees camera timing even though software WebGL is
far slower. `--motion-zoom 0.6` shrinks the faces (a person further from the camera).

The ground truth is the still start's tracked points moved by the known transform. Metrics (see
`motion.cjs`; distances are a share of the eye distance over all 468 mesh points):
`lag` (the smooth part of the error: smoothing delay plus the tracker's own), `jitter` (frame-to-frame
noise), `jitterHold` (noise while the head is still), `raw*` (the same for MediaPipe's points before the
engine's smoothing), `lost` (frames without tracking), `popMax` (largest one-frame change in face
opacity), `backErr`/`backFade` (after the dropout), and identity on the still start (`idRef0`) and on 4
moving frames (`idRefFrames`, `idRefFramesMin`, `poseFrames`). `--images N` writes `motion.jpg`.
Each pair's tracks are kept in `<out>/<target>.motion.json`.

MediaPipe's raw points don't depend on the smoothing, so smoothing settings (`minCutoff`, `beta`,
`dCutoff`, `holdMs`, `fadeInMs`, `fadeOutMs`) can be re-scored offline on recorded tracks in about a
second: `node scripts/bench/motion-replay.mjs <out dir> --grid '[{"beta":6},{"beta":12}]'`. It runs the
engine's own `PointSmoother`, so it matches a full `--motion` run's landmark metrics (identity needs a
real run).
