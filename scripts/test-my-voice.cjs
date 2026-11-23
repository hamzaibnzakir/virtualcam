'use strict';
// "My voices" check: prepares a voice from a recording, then streams another recording through it live.
//   node scripts/test-my-voice.cjs <modelsDir> <voice-sample.wav (16 kHz)> <speech.wav (48 kHz)> <out.wav> [whole-out.wav]
const os = require('node:os');
const path = require('node:path');
const ort = require('onnxruntime-node');
const { MatchConverter } = require('../src/main/voice/match');
const { readWav, writeWav } = require('./test-voice.cjs');

(async () => {
  const [modelsDir, sample, input, output, whole] = process.argv.slice(2);
  const voicesDir = path.join(os.tmpdir(), 'bbx-voices-test');
  const mc = new MatchConverter(ort, modelsDir, voicesDir, { log: (m) => console.log(m) });
  const loaded = await mc.load({});
  const ref = readWav(sample);
  if (ref.rate !== 16000) throw new Error('voice sample must be 16 kHz');
  let t = Date.now();
  const prepared = await mc.prepare('test-voice', ref.samples, (p) => process.stdout.write(`\rpreparing ${Math.round(p * 100)}%`));
  console.log(`\nprepared in ${(Date.now() - t) / 1000}s`, JSON.stringify(prepared));
  await mc.setVoice('test-voice');
  const { samples } = readWav(input);
  const out = [];
  mc.onAudio = (pcm) => out.push(pcm);
  t = Date.now();
  for (let i = 0; i < samples.length; i += 960) {
    mc.push(samples.subarray(i, i + 960));
    while (mc.busy) await new Promise((r) => setTimeout(r, 1));
  }
  const all = new Float32Array(out.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of out) {
    all.set(a, o);
    o += a.length;
  }
  writeWav(output, all, 48000);
  console.log(JSON.stringify({ loaded, inSeconds: samples.length / 48000, outSeconds: all.length / 48000, wallSeconds: (Date.now() - t) / 1000, stats: mc.stats() }));
  if (whole) {
    const { Resampler } = require('../src/main/voice/dsp');
    const pcm16 = new Resampler(48000, 16000).push(samples);
    t = Date.now();
    const w = await mc.convertWhole(pcm16);
    writeWav(whole, w, 16000);
    console.log(`whole-file conversion ${(Date.now() - t) / 1000}s`);
  }
  await mc.release();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
