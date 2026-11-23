'use strict';

// Live AI voice conversion (RVC v2). Your voice goes in at 48 kHz in small chunks; the chosen trained
// voice comes out at 48 kHz, about a quarter of a second later.
//
// Every block:
//   1. the last ~1.3 s of your voice at 16 kHz (new block + context before it)
//   2. ContentVec turns it into "what is being said" features (50 per second, doubled to 100)
//   3. RMVPE finds your pitch (100 per second), moved into the voice's own range
//   4. the voice model speaks those words at that pitch in its own voice (40 kHz), only the newest part
//   5. the new part is lined up with the previous one (SOLA) and crossfaded, so blocks join smoothly
//   6. 40 kHz -> 48 kHz
const fs = require('node:fs');
const path = require('node:path');
const { Resampler, MelSpectrogram, decodePitch, coarsePitch, median, rms } = require('./dsp');

const SR = 16000;
const FRAME = 160; // 10 ms at 16 kHz: one pitch / feature frame
const CONTEXT = 16000; // 1 s of context before each block
const CROSS = 800; // 50 ms crossfade
const SEARCH = 160; // 10 ms to search for the best join
const BLOCK_STEPS = [4000, 6400, 8000]; // 0.25 s, 0.4 s, 0.5 s; longer only if the device is slow
const SILENCE = 0.004; // below this loudness (about -48 dB) a block is silence: skip the AI

const CONTENT_FILE = 'contentvec.onnx';
const PITCH_FILE = 'rmvpe.onnx';

function sessionOptions(device) {
  const options = {
    executionProviders: [device.name === 'dml' ? { name: 'dml', deviceId: device.deviceId } : device.name],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3,
  };
  if (device.name === 'dml') {
    options.enableMemPattern = false;
    options.executionMode = 'sequential';
  }
  return options;
}

function deviceCandidates(preferred, gpuCount = 0) {
  if (preferred === 'cpu') return [{ name: 'cpu' }];
  if (process.platform === 'win32') {
    const gpus = gpuCount === 1 ? [0] : [0, 1];
    return [...gpus.map((deviceId) => ({ name: 'dml', deviceId })), { name: 'cpu' }];
  }
  return [{ name: 'cpu' }];
}

function deviceLabel(device) {
  if (device.name === 'dml') return `graphics chip ${device.deviceId + 1}`;
  return 'the processor';
}

class VoiceConverter {
  constructor(ort, modelsDir, { log = () => {} } = {}) {
    this.ort = ort;
    this.modelsDir = modelsDir;
    this.log = log;
    this.device = null;
    this.content = null;
    this.pitch = null;
    this.voices = new Map(); // file -> session
    this.voice = null; // { session, sr, f0, file }
    this.mel = new MelSpectrogram();
    this.onAudio = () => {};
    this.onInfo = () => {};
    this.block = BLOCK_STEPS[0];
    this.speakerF0 = 0;
    this.reset();
  }

  reset() {
    this.down = new Resampler(48000, SR);
    this.up = this.voice ? new Resampler(this.voice.sr, 48000) : null;
    this.pending = []; // 16 kHz samples waiting to make a block
    this.pendingLength = 0;
    this.window = new Float32Array(CONTEXT + this.block + CROSS + SEARCH);
    this.sola = null; // the tail of the previous block, to join the next one to
    this.busy = false;
    this.speakerPitches = [];
    this.speakerF0 = this.speakerF0 || 0;
    this.slowSteps = 0;
    this.steps = 0;
    this.stepMs = 0;
  }

  get ready() {
    return Boolean(this.content && this.pitch && this.voice);
  }

  // Loads the two shared models on the fastest device (graphics chip if it works, else processor).
  async load({ preferred = 'auto', gpuCount = 0, onStep = () => {} } = {}) {
    const file = (f) => path.join(this.modelsDir, f);
    for (const f of [CONTENT_FILE, PITCH_FILE]) if (!fs.existsSync(file(f))) throw new Error(`Missing voice model file ${f}`);
    const tried = [];
    let best = null;
    for (const device of deviceCandidates(preferred, gpuCount)) {
      if (device.name === 'cpu' && best && best.ms < 120) break;
      const label = deviceLabel(device);
      try {
        onStep(`loading on ${label}`);
        const content = await this.ort.InferenceSession.create(file(CONTENT_FILE), sessionOptions(device));
        const pitch = await this.ort.InferenceSession.create(file(PITCH_FILE), sessionOptions(device));
        onStep(`timing ${label}`);
        const ms = await this.timeShared(content, pitch);
        tried.push({ device: label, ms });
        this.log(`voice engine: ${label} takes ${ms} ms per block`);
        if (!best || ms < best.ms) {
          if (best) await Promise.all([best.content.release(), best.pitch.release()]).catch(() => {});
          best = { device, content, pitch, ms };
        } else {
          await Promise.all([content.release(), pitch.release()]).catch(() => {});
        }
      } catch (err) {
        tried.push({ device: label, error: (err && err.message) || String(err) });
        this.log(`voice engine: ${label} failed: ${(err && err.message) || err}`);
      }
    }
    if (!best) throw new Error(`The voice models could not start: ${tried.map((t) => `${t.device}: ${t.error}`).join('; ')}`);
    this.device = best.device;
    this.content = best.content;
    this.pitch = best.pitch;
    this.voices.clear();
    this.voice = null;
    return { device: deviceLabel(best.device), ms: best.ms, tried };
  }

  async timeShared(content, pitch) {
    const audio = new Float32Array(this.window.length).map((_, i) => 0.1 * Math.sin(i * 0.07));
    const run = async () => {
      await this.features(audio, content);
      await this.findPitch(audio, pitch);
    };
    await run();
    const t = Date.now();
    await run();
    return Date.now() - t;
  }

  // voice: { file, sr, f0 } from the voice pack manifest. f0 is the voice's typical speaking pitch.
  async setVoice(voice) {
    if (!this.content) throw new Error('The voice engine is not loaded');
    if (!voice || !/^[\w.-]+\.onnx$/.test(voice.file)) throw new Error('Unknown voice');
    let session = this.voices.get(voice.file);
    if (!session) {
      const file = path.join(this.modelsDir, voice.file);
      if (!fs.existsSync(file)) throw new Error(`Missing voice file ${voice.file}`);
      session = await this.ort.InferenceSession.create(file, sessionOptions(this.device));
      this.voices.set(voice.file, session);
    }
    const sr = Number(voice.sr) || 40000;
    this.voice = { session, sr, f0: Number(voice.f0) || 0, file: voice.file };
    this.up = new Resampler(sr, 48000);
    this.sola = null;
    // Warm up (the first run on a graphics chip sets things up and can be slow).
    await this.convert(this.window);
    return { file: voice.file, sr };
  }

  async features(audio, session = this.content) {
    const out = await session.run({ audio: new this.ort.Tensor('float32', audio, [1, audio.length]) });
    const feats = out.feats || Object.values(out)[0];
    return { data: feats.data, frames: feats.dims[1] };
  }

  async findPitch(audio, session = this.pitch) {
    const { data, frames } = this.mel.compute(audio);
    // RMVPE wants a multiple of 32 frames; pad with zeros (as RVC does) and keep the real ones.
    const padded = 32 * Math.ceil(frames / 32);
    const input = new Float32Array(128 * padded);
    for (let m = 0; m < 128; m += 1) input.set(data.subarray(m * frames, (m + 1) * frames), m * padded);
    const out = await session.run({ input: new this.ort.Tensor('float32', input, [1, 128, padded]) });
    const sal = out.output || Object.values(out)[0];
    // [1, T, 360] (or [T, 360])
    return decodePitch(sal.data, frames);
  }

  // How far to move your pitch so it lands in the voice's range (in semitones, at most an octave).
  shiftFor() {
    if (!this.voice || !this.voice.f0 || !this.speakerF0) return 0;
    const st = 12 * Math.log2(this.voice.f0 / this.speakerF0);
    return Math.max(-12, Math.min(12, st));
  }

  // Runs the AI on a window of 16 kHz audio and returns the newest `outFrames` frames at the voice's rate.
  async convert(audio, outFrames = Math.round((this.block + CROSS + SEARCH) / FRAME)) {
    const pLen = Math.floor(audio.length / FRAME);
    const [{ data: feats, frames: t50 }, f0] = await Promise.all([this.features(audio), this.findPitch(audio)]);
    // Features: 50 per second -> 100 per second (each repeated), padded/cut to one per 10 ms frame.
    const phone = new Float32Array(pLen * 768);
    for (let i = 0; i < pLen; i += 1) {
      const src = Math.min(t50 - 1, Math.floor(i / 2));
      phone.set(feats.subarray(src * 768, src * 768 + 768), i * 768);
    }
    // Learn the speaker's own pitch (voiced frames of the newest part).
    for (let i = pLen - outFrames; i < pLen; i += 1) if (f0[i] > 60 && f0[i] < 500) this.speakerPitches.push(f0[i]);
    if (this.speakerPitches.length > 3000) this.speakerPitches.splice(0, this.speakerPitches.length - 3000);
    if (this.speakerPitches.length >= 50) {
      const m = median(this.speakerPitches);
      if (!this.speakerF0 || Math.abs(m - this.speakerF0) / this.speakerF0 > 0.03) {
        this.speakerF0 = m;
        this.onInfo({ speakerF0: Math.round(m), shift: Number(this.shiftFor().toFixed(1)) });
      }
    }
    const factor = 2 ** (this.shiftFor() / 12);
    const nsf = new Float32Array(pLen);
    for (let i = 0; i < pLen; i += 1) nsf[i] = f0[i] * factor;
    const { session, sr } = this.voice;
    const rate = outFrames / pLen;
    const feeds = {
      phone: new this.ort.Tensor('float32', phone, [1, pLen, 768]),
      phone_lengths: new this.ort.Tensor('int64', BigInt64Array.from([BigInt(pLen)]), [1]),
      sid: new this.ort.Tensor('int64', BigInt64Array.from([0n]), [1]),
      rate: new this.ort.Tensor('float32', Float32Array.from([rate]), [1]),
      pitch: new this.ort.Tensor('int64', coarsePitch(nsf), [1, pLen]),
      nsff0: new this.ort.Tensor('float32', nsf, [1, pLen]),
    };
    const out = await session.run(feeds);
    const audioOut = (out.audio || Object.values(out)[0]).data;
    const want = outFrames * (sr / 100);
    return audioOut.length >= want ? audioOut.subarray(audioOut.length - want) : audioOut;
  }

  // Feed your voice (48 kHz mono, any chunk size). Converted audio arrives through onAudio.
  push(pcm48) {
    const s16 = this.down.push(pcm48);
    if (s16.length) {
      this.pending.push(s16);
      this.pendingLength += s16.length;
    }
    // Never let more than ~1 s pile up behind a slow device: drop the oldest.
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
        // Slide the window along and add the new block at the end.
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
      this.log(`voice engine step failed: ${(err && err.stack) || err}`);
      this.onInfo({ error: (err && err.message) || String(err) });
    } finally {
      this.busy = false;
    }
  }

  // If the device can't keep up, make blocks longer (a little more delay, but no gaps).
  adapt(ms) {
    const budget = (this.block / SR) * 1000;
    if (ms > budget * 0.85) this.slowSteps += 1;
    else this.slowSteps = Math.max(0, this.slowSteps - 1);
    const i = BLOCK_STEPS.indexOf(this.block);
    if (this.slowSteps >= 4 && i < BLOCK_STEPS.length - 1) {
      this.setBlock(BLOCK_STEPS[i + 1]);
      this.log(`voice engine: blocks now ${Math.round((this.block / SR) * 1000)} ms (a block took ${ms} ms)`);
      this.onInfo({ blockMs: Math.round((this.block / SR) * 1000) });
    }
  }

  setBlock(samples) {
    const old = this.window;
    this.block = samples;
    this.window = new Float32Array(CONTEXT + this.block + CROSS + SEARCH);
    const keep = Math.min(old.length, this.window.length);
    this.window.set(old.subarray(old.length - keep), this.window.length - keep);
    this.slowSteps = 0;
  }

  // One block: returns `block` samples at the voice's rate (silence for quiet blocks).
  async step(block) {
    const { sr } = this.voice;
    const ratio = sr / SR;
    const blockOut = Math.round(this.block * ratio);
    const crossOut = Math.round(CROSS * ratio);
    const searchOut = Math.round(SEARCH * ratio);
    const loud = rms(block) >= SILENCE || rms(this.window, this.window.length - this.block - CROSS) >= SILENCE;
    if (!loud) {
      // Finish the previous sound smoothly, then silence.
      const out = new Float32Array(blockOut);
      if (this.sola) for (let i = 0; i < crossOut; i += 1) out[i] = this.sola[i] * (1 - i / crossOut);
      this.sola = null;
      return out;
    }
    const audio = await this.convert(this.window);
    // Keep the loudness of your own voice (the model's own level varies).
    const inLevel = rms(this.window, this.window.length - this.block - CROSS - SEARCH);
    const outLevel = rms(audio);
    const gain = outLevel > 1e-5 ? Math.max(0.25, Math.min(4, inLevel / outLevel)) ** 0.75 : 1;
    if (audio.length < searchOut + blockOut + crossOut) return null;
    let offset = 0;
    if (this.sola) {
      // Find where the new audio best continues the old tail.
      let bestScore = -Infinity;
      for (let o = 0; o <= searchOut; o += 1) {
        let num = 0;
        let den = 1e-8;
        for (let i = 0; i < crossOut; i += 1) {
          num += audio[o + i] * this.sola[i];
          den += audio[o + i] * audio[o + i];
        }
        const score = num / Math.sqrt(den);
        if (score > bestScore) {
          bestScore = score;
          offset = o;
        }
      }
    }
    const out = new Float32Array(blockOut);
    for (let i = 0; i < blockOut; i += 1) out[i] = audio[offset + i] * gain;
    if (this.sola) {
      for (let i = 0; i < crossOut; i += 1) {
        const w = Math.sin((0.5 * Math.PI * i) / crossOut) ** 2;
        out[i] = out[i] * w + this.sola[i] * (1 - w);
      }
    } else {
      for (let i = 0; i < crossOut; i += 1) out[i] *= Math.sin((0.5 * Math.PI * i) / crossOut) ** 2;
    }
    this.sola = Float32Array.from(audio.subarray(offset + blockOut, offset + blockOut + crossOut), (v) => v * gain);
    for (let i = 0; i < out.length; i += 1) out[i] = Math.max(-1, Math.min(1, out[i]));
    return out;
  }

  stats() {
    return {
      device: this.device ? deviceLabel(this.device) : null,
      steps: this.steps,
      blockMs: Math.round((this.block / SR) * 1000),
      stepMs: Math.round(this.stepMs),
      speakerF0: Math.round(this.speakerF0),
      shift: Number(this.shiftFor().toFixed(1)),
      // mic -> you hear it: one block to collect + the work + the crossfade/search margin
      delayMs: Math.round(((this.block + CROSS + SEARCH) / SR) * 1000 + this.stepMs),
    };
  }

  async release() {
    const all = [this.content, this.pitch, ...this.voices.values()].filter(Boolean);
    await Promise.all(all.map((s) => s.release().catch(() => {})));
    this.content = null;
    this.pitch = null;
    this.voices.clear();
    this.voice = null;
  }
}

module.exports = { VoiceConverter, CONTENT_FILE, PITCH_FILE };
