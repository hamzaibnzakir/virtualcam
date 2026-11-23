'use strict';

// "My voices": a voice made from your own upload, with no training (kNN-VC).
//
// Preparing a voice (once, about a minute): WavLM describes every 20 ms of the upload; those
// descriptions are the voice. Live, every block:
//   1. the last ~1 s of your voice at 16 kHz -> WavLM descriptions of what you're saying
//   2. each is replaced by the average of the 4 most similar moments of the uploaded voice
//   3. the vocoder turns that into sound (16 kHz), joined to the previous block, then 16 -> 48 kHz
const fs = require('node:fs');
const path = require('node:path');
const { Resampler, Joiner, rms } = require('./dsp');
const { sessionOptions, deviceCandidates, deviceLabel } = require('./converter');

const SR = 16000;
const HOP = 320; // one description per 20 ms
const WIN = 400; // each description looks at 25 ms
const DIM = 1024;
const CONTEXT_FRAMES = 40; // 0.8 s before each block
const LOOK = 2; // 40 ms held back so the newest sound has a little "after" context too
const VOC_CONTEXT = 8; // frames the vocoder hears before the part we keep
const CROSS = 640; // 40 ms crossfade
const SEARCH = 160;
const BLOCK_STEPS = [12, 16, 20]; // frames per block: 0.24 s, 0.32 s, 0.4 s
const SILENCE = 0.004;

const FILES = { features: 'wavlm6.onnx', vocoder: 'knn-vocoder.onnx', match: 'knn-match.onnx' };
const MAGIC = 0x56584242; // "BBXV"
const MIN_SECONDS = 30;

function toHalf(v) {
  const f = new Float32Array([v]);
  const x = new Uint32Array(f.buffer)[0];
  const sign = (x >>> 16) & 0x8000;
  let exp = ((x >>> 23) & 0xff) - 127 + 15;
  const mant = x & 0x7fffff;
  if (exp <= 0) return sign;
  if (exp >= 31) return sign | 0x7c00;
  return sign | (exp << 10) | (mant >>> 13);
}

function fromHalf(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * 2 ** -14 * (mant / 1024);
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * 2 ** (exp - 15) * (1 + mant / 1024);
}

// A prepared voice on disk: header (magic, version, frames, dim) + float16 descriptions.
function writeVoiceFile(file, feats, frames) {
  const buf = Buffer.alloc(16 + frames * DIM * 2);
  buf.writeUInt32LE(MAGIC, 0);
  buf.writeUInt32LE(1, 4);
  buf.writeUInt32LE(frames, 8);
  buf.writeUInt32LE(DIM, 12);
  for (let i = 0; i < frames * DIM; i += 1) buf.writeUInt16LE(toHalf(feats[i]), 16 + i * 2);
  fs.writeFileSync(file, buf);
}

function readVoiceFile(file) {
  const buf = fs.readFileSync(file);
  if (buf.readUInt32LE(0) !== MAGIC || buf.readUInt32LE(12) !== DIM) throw new Error('That voice file is damaged. Prepare the voice again.');
  const frames = buf.readUInt32LE(8);
  const lut = new Float32Array(65536);
  for (let h = 0; h < 65536; h += 1) lut[h] = fromHalf(h);
  const out = new Float32Array(frames * DIM);
  for (let i = 0; i < out.length; i += 1) out[i] = lut[buf.readUInt16LE(16 + i * 2)];
  return { feats: out, frames };
}

class MatchConverter {
  constructor(ort, modelsDir, voicesDir, { log = () => {} } = {}) {
    this.ort = ort;
    this.modelsDir = modelsDir;
    this.voicesDir = voicesDir;
    this.log = log;
    this.device = null;
    this.features = null;
    this.vocoder = null;
    this.match = null;
    this.voice = null; // { id, set: ort.Tensor }
    this.onAudio = () => {};
    this.onInfo = () => {};
    this.blockFrames = BLOCK_STEPS[0];
    this.reset();
  }

  static files() {
    return Object.values(FILES);
  }

  get block() {
    return this.blockFrames * HOP;
  }

  get windowLength() {
    return (CONTEXT_FRAMES + this.blockFrames + LOOK - 1) * HOP + WIN;
  }

  reset() {
    this.down = new Resampler(48000, SR);
    this.up = new Resampler(SR, 48000);
    this.joiner = new Joiner(CROSS, SEARCH);
    this.pending = [];
    this.pendingLength = 0;
    this.window = new Float32Array(this.windowLength);
    this.busy = false;
    this.slowSteps = 0;
    this.steps = 0;
    this.stepMs = 0;
  }

  get ready() {
    return Boolean(this.features && this.vocoder && this.match && this.voice);
  }

  async load({ preferred = 'auto', gpuCount = 0, onStep = () => {} } = {}) {
    const file = (k) => path.join(this.modelsDir, FILES[k]);
    for (const k of Object.keys(FILES)) if (!fs.existsSync(file(k))) throw new Error(`Missing voice model file ${FILES[k]}`);
    // Matching works on the processor (its voice data stays in memory, no copying to the graphics chip).
    this.match = await this.ort.InferenceSession.create(file('match'), sessionOptions({ name: 'cpu' }));
    const tried = [];
    let best = null;
    for (const device of deviceCandidates(preferred, gpuCount)) {
      if (device.name === 'cpu' && best && best.ms < 120) break;
      const label = deviceLabel(device);
      try {
        onStep(`loading on ${label}`);
        const features = await this.ort.InferenceSession.create(file('features'), sessionOptions(device));
        const vocoder = await this.ort.InferenceSession.create(file('vocoder'), sessionOptions(device));
        const ms = await this.timeModels(features, vocoder);
        tried.push({ device: label, ms });
        this.log(`my voices: ${label} takes ${ms} ms per block`);
        if (!best || ms < best.ms) {
          if (best) await Promise.all([best.features.release(), best.vocoder.release()]).catch(() => {});
          best = { device, features, vocoder, ms };
        } else await Promise.all([features.release(), vocoder.release()]).catch(() => {});
      } catch (err) {
        tried.push({ device: label, error: (err && err.message) || String(err) });
        this.log(`my voices: ${label} failed: ${(err && err.message) || err}`);
      }
    }
    if (!best) throw new Error(`The voice models could not start: ${tried.map((t) => `${t.device}: ${t.error}`).join('; ')}`);
    Object.assign(this, { device: best.device, features: best.features, vocoder: best.vocoder });
    return { device: deviceLabel(best.device), ms: best.ms, tried };
  }

  async timeModels(features, vocoder) {
    const audio = new Float32Array(this.windowLength).map((_, i) => 0.1 * Math.sin(i * 0.07));
    const run = async () => {
      const f = await this.describe(audio, features);
      const n = this.blockFrames + VOC_CONTEXT + 3;
      await this.speak(f.data.subarray((f.frames - n) * DIM), n, vocoder);
    };
    await run();
    const t = Date.now();
    await run();
    return Date.now() - t;
  }

  async describe(audio, session = this.features) {
    const out = await session.run({ audio: new this.ort.Tensor('float32', audio, [1, audio.length]) });
    const t = out.feats || Object.values(out)[0];
    return { data: t.data, frames: t.dims[1] };
  }

  async speak(feats, frames, session = this.vocoder) {
    const out = await session.run({ feats: new this.ort.Tensor('float32', Float32Array.from(feats.subarray(0, frames * DIM)), [1, frames, DIM]) });
    return (out.audio || Object.values(out)[0]).data;
  }

  async nearest(query, frames) {
    const out = await this.match.run({ query: new this.ort.Tensor('float32', query, [frames, DIM]), set: this.voice.set });
    return (out.matched || Object.values(out)[0]).data;
  }

  // ---------- preparing a voice from an upload ----------

  // pcm: 16 kHz mono Float32Array. Saves <id>.voice in the voices folder. onProgress(0..1).
  async prepare(id, pcm, onProgress = () => {}) {
    if (!this.features) throw new Error('The voice engine is not loaded');
    if (!/^[\w-]{1,64}$/.test(id)) throw new Error('Bad voice id');
    const chunk = 10 * SR;
    const kept = [];
    let frames = 0;
    let speech = 0;
    const levels = [];
    // Loudness of each 20 ms, to leave out silence (pauses would otherwise turn into gaps).
    for (let i = 0; i + WIN <= pcm.length; i += HOP) levels.push(rms(pcm, i, i + WIN));
    const sorted = [...levels].sort((a, b) => a - b);
    const loudish = sorted[Math.floor(sorted.length * 0.9)] || 0;
    const floor = Math.max(0.003, loudish * 0.08);
    for (let start = 0; start < pcm.length; start += chunk) {
      const piece = pcm.subarray(start, Math.min(pcm.length, start + chunk));
      if (piece.length < WIN * 4) break;
      const { data, frames: n } = await this.describe(Float32Array.from(piece));
      const base = Math.floor(start / HOP);
      for (let f = 0; f < n; f += 1) {
        if ((levels[base + f] || 0) < floor) continue;
        kept.push(data.subarray(f * DIM, (f + 1) * DIM));
        frames += 1;
      }
      onProgress(Math.min(1, (start + chunk) / pcm.length));
    }
    speech = (frames * HOP) / SR;
    if (speech < MIN_SECONDS) {
      throw new Error(`Only ${Math.round(speech)} seconds of speech found. Use at least ${MIN_SECONDS} seconds (2-5 minutes sounds much better).`);
    }
    const feats = new Float32Array(frames * DIM);
    kept.forEach((row, i) => feats.set(row, i * DIM));
    fs.mkdirSync(this.voicesDir, { recursive: true });
    const file = path.join(this.voicesDir, `${id}.voice`);
    writeVoiceFile(`${file}.part`, feats, frames);
    fs.renameSync(`${file}.part`, file);
    return { id, frames, seconds: Math.round(speech), quality: speech >= 180 ? 'great' : speech >= 90 ? 'good' : 'basic' };
  }

  async setVoice(id) {
    if (!this.features) throw new Error('The voice engine is not loaded');
    if (!/^[\w-]{1,64}$/.test(id)) throw new Error('Unknown voice');
    if (this.voice && this.voice.id === id) return { id };
    const { feats, frames } = readVoiceFile(path.join(this.voicesDir, `${id}.voice`));
    this.voice = { id, set: new this.ort.Tensor('float32', feats, [frames, DIM]) };
    this.joiner.reset();
    return { id, frames };
  }

  // ---------- live ----------

  push(pcm48) {
    const s16 = this.down.push(pcm48);
    if (s16.length) {
      this.pending.push(s16);
      this.pendingLength += s16.length;
    }
    const max = this.block * 4;
    while (this.pendingLength > max && this.pending.length > 1) this.pendingLength -= this.pending.shift().length;
    if (!this.busy) this.pump();
  }

  takeBlock() {
    const out = new Float32Array(this.block);
    let n = 0;
    while (n < this.block) {
      const head = this.pending[0];
      const take = Math.min(head.length, this.block - n);
      out.set(head.subarray(0, take), n);
      n += take;
      if (take === head.length) this.pending.shift();
      else this.pending[0] = head.subarray(take);
    }
    this.pendingLength -= this.block;
    return out;
  }

  async pump() {
    this.busy = true;
    try {
      while (this.ready && this.pendingLength >= this.block) {
        const block = this.takeBlock();
        this.window.copyWithin(0, this.block);
        this.window.set(block, this.window.length - this.block);
        const t = Date.now();
        const out = await this.step(block);
        const ms = Date.now() - t;
        this.steps += 1;
        this.stepMs = this.stepMs ? this.stepMs * 0.9 + ms * 0.1 : ms;
        if (out) {
          const pcm = this.up.push(out);
          if (pcm.length) this.onAudio(pcm);
        }
        this.adapt(ms);
      }
    } catch (err) {
      this.log(`my voices step failed: ${(err && err.stack) || err}`);
      this.onInfo({ error: (err && err.message) || String(err) });
    } finally {
      this.busy = false;
    }
  }

  adapt(ms) {
    if (ms > (this.block / SR) * 1000 * 0.85) this.slowSteps += 1;
    else this.slowSteps = Math.max(0, this.slowSteps - 1);
    const i = BLOCK_STEPS.indexOf(this.blockFrames);
    if (this.slowSteps >= 4 && i < BLOCK_STEPS.length - 1) {
      const old = this.window;
      this.blockFrames = BLOCK_STEPS[i + 1];
      this.window = new Float32Array(this.windowLength);
      const keep = Math.min(old.length, this.window.length);
      this.window.set(old.subarray(old.length - keep), this.window.length - keep);
      this.slowSteps = 0;
      this.log(`my voices: blocks now ${Math.round((this.block / SR) * 1000)} ms (a block took ${ms} ms)`);
      this.onInfo({ blockMs: Math.round((this.block / SR) * 1000) });
    }
  }

  async step(block) {
    const loud = rms(block) >= SILENCE || rms(this.window, this.window.length - this.block - CROSS) >= SILENCE;
    if (!loud) return this.joiner.silence(this.block);
    const { data, frames } = await this.describe(this.window);
    // The frames to convert: the newest block (minus the held-back LOOK), plus the joining margin, plus
    // the vocoder's own context in front.
    const keep = this.blockFrames + 3; // block + cross (2 frames) + search margin
    const n = keep + VOC_CONTEXT;
    const end = frames - LOOK;
    const start = Math.max(0, end - n);
    const query = Float32Array.from(data.subarray(start * DIM, end * DIM));
    const count = end - start;
    const matched = await this.nearest(query, count);
    const audio = await this.speak(matched, count);
    const tail = audio.subarray(Math.max(0, audio.length - keep * HOP));
    const inLevel = rms(this.window, this.window.length - this.block - CROSS - SEARCH);
    const outLevel = rms(tail);
    const gain = outLevel > 1e-5 ? Math.max(0.25, Math.min(4, inLevel / outLevel)) ** 0.75 : 1;
    return this.joiner.join(tail, this.block, gain);
  }

  stats() {
    return {
      engine: 'my-voice',
      device: this.device ? deviceLabel(this.device) : null,
      steps: this.steps,
      blockMs: Math.round((this.block / SR) * 1000),
      stepMs: Math.round(this.stepMs),
      delayMs: Math.round(((this.block + CROSS + SEARCH + LOOK * HOP) / SR) * 1000 + this.stepMs),
    };
  }

  // Whole recording at once (voice messages): best quality, nothing streamed.
  async convertWhole(pcm16) {
    const { data, frames } = await this.describe(pcm16);
    const matched = await this.nearest(Float32Array.from(data), frames);
    return this.speak(matched, frames);
  }

  async release() {
    const all = [this.features, this.vocoder, this.match].filter(Boolean);
    await Promise.all(all.map((s) => s.release().catch(() => {})));
    this.features = null;
    this.vocoder = null;
    this.match = null;
    this.voice = null;
  }
}

module.exports = { MatchConverter, MATCH_FILES: FILES, readVoiceFile, writeVoiceFile };
