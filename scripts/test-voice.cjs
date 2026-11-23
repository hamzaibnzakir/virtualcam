'use strict';
// Streams a recording through the app's AI voice engine, the way the microphone feeds it live, and
// writes what comes out.
//   node scripts/test-voice.cjs <modelsDir> <voice.onnx> <in.wav (48 kHz mono)> <out.wav> [voiceF0] [sr]
const fs = require('node:fs');
const path = require('node:path');
const ort = require('onnxruntime-node');
const { VoiceConverter } = require('../src/main/voice/converter');

function readWav(file) {
  const b = fs.readFileSync(file);
  let p = 12;
  let fmt = null;
  while (p < b.length) {
    const id = b.toString('ascii', p, p + 4);
    const size = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { format: b.readUInt16LE(p + 8), channels: b.readUInt16LE(p + 10), rate: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') {
      const n = size / (fmt.bits / 8) / fmt.channels;
      const out = new Float32Array(n);
      for (let i = 0; i < n; i += 1) {
        const o = p + 8 + i * fmt.channels * (fmt.bits / 8);
        out[i] = fmt.format === 3 ? b.readFloatLE(o) : fmt.bits === 16 ? b.readInt16LE(o) / 32768 : b.readInt32LE(o) / 2147483648;
      }
      return { rate: fmt.rate, samples: out };
    }
    p += 8 + size + (size % 2);
  }
  throw new Error('no data chunk');
}

function writeWav(file, samples, rate) {
  const b = Buffer.alloc(44 + samples.length * 2);
  b.write('RIFF', 0);
  b.writeUInt32LE(36 + samples.length * 2, 4);
  b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16);
  b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24);
  b.writeUInt32LE(rate * 2, 28);
  b.writeUInt16LE(2, 32);
  b.writeUInt16LE(16, 34);
  b.write('data', 36);
  b.writeUInt32LE(samples.length * 2, 40);
  for (let i = 0; i < samples.length; i += 1) b.writeInt16LE(Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), 44 + i * 2);
  fs.writeFileSync(file, b);
}

async function main() {
  const [modelsDir, voiceFile, input, output, f0 = '0', sr = '40000'] = process.argv.slice(2);
  const { rate, samples } = readWav(input);
  if (rate !== 48000) throw new Error('input must be 48 kHz');
  const vc = new VoiceConverter(ort, modelsDir, { log: (m) => console.log(m) });
  const loaded = await vc.load({});
  await vc.setVoice({ file: path.basename(voiceFile), sr: Number(sr), f0: Number(f0) });
  const out = [];
  vc.onAudio = (pcm) => out.push(pcm);
  vc.onInfo = (info) => console.log('info', JSON.stringify(info));
  const chunk = 960; // 20 ms, as the studio sends it
  const t = Date.now();
  // Real time: feed one chunk every 20 ms of audio, but don't wait (measure whether it keeps up).
  for (let i = 0; i < samples.length; i += chunk) {
    vc.push(samples.subarray(i, i + chunk));
    while (vc.busy) await new Promise((r) => setTimeout(r, 1));
  }
  const ms = Date.now() - t;
  const all = new Float32Array(out.reduce((s, a) => s + a.length, 0));
  let o = 0;
  for (const a of out) {
    all.set(a, o);
    o += a.length;
  }
  writeWav(output, all, 48000);
  console.log(JSON.stringify({ loaded, inSeconds: samples.length / 48000, outSeconds: all.length / 48000, wallSeconds: ms / 1000, stats: vc.stats() }));
  await vc.release();
}

module.exports = { readWav, writeWav };
if (require.main === module) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
