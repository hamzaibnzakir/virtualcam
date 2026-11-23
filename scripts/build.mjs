// Bundles the renderer code (studio, switcher) and copies the static files into dist/.
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'dist');

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

await build({
  entryPoints: {
    studio: join(root, 'src/renderer/studio.js'),
    switcher: join(root, 'src/renderer/switcher.js'),
    prompt: join(root, 'src/renderer/prompt.js'),
  },
  bundle: true,
  format: 'iife',
  platform: 'browser',
  target: 'chrome130',
  outdir: out,
  sourcemap: true,
  minify: true,
  logLevel: 'warning',
  legalComments: 'none',
});

for (const file of ['studio.html', 'switcher.html', 'prompt.html', 'styles.css']) {
  cpSync(join(root, 'src/renderer', file), join(out, file));
}

// Lite engine: MediaPipe's WebAssembly runtime, the face tracking model and the hair/skin segmentation
// model ship with the app.
mkdirSync(join(out, 'mediapipe'), { recursive: true });
for (const file of ['vision_wasm_internal.js', 'vision_wasm_internal.wasm']) {
  cpSync(join(root, 'node_modules/@mediapipe/tasks-vision/wasm', file), join(out, 'mediapipe', file));
}
for (const file of ['face_landmarker.task', 'selfie_multiclass_256x256.tflite']) {
  cpSync(join(root, 'assets/mediapipe', file), join(out, 'mediapipe', file));
}

// Brand images (icon on the prompt card and in the studio).
mkdirSync(join(out, 'brand'), { recursive: true });
cpSync(join(root, 'assets/brand/icon.png'), join(out, 'brand', 'icon.png'));

console.log('Built renderer into dist/');
