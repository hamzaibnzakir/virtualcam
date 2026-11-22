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

for (const file of ['studio.html', 'switcher.html', 'styles.css']) {
  cpSync(join(root, 'src/renderer', file), join(out, file));
}

console.log('Built renderer into dist/');
