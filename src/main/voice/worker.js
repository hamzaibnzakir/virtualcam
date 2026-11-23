'use strict';

// Voice engine process: the AI voices, run away from the app window. Two kinds of voice:
//   trained  the four preset voices (RVC, trained ahead of time)            -> converter.js
//   mine     voices made from your own uploads (kNN-VC, ready in a minute)  -> match.js
// Each engine loads the first time a voice of its kind is used.
//
// Two ports from the studio:
//   control: -> { type: 'load', modelsDir, voicesDir, preferred, gpuCount }  <- { type: 'loaded' }
//            -> { type: 'voice', voice: { kind: 'trained', file, sr, f0 } | { kind: 'mine', id } }
//                                                       <- { type: 'voice', kind, device, ms } | { type: 'error', request, message }
//            -> { type: 'prepare', id, pcm (16 kHz) }   <- { type: 'prepare-progress', id, progress } ... { type: 'prepared', id, seconds, quality }
//            -> { type: 'convert', voice, pcm (16 kHz) } <- { type: 'converted', pcm, rate }   (a whole recording: voice messages)
//            -> { type: 'stats' } <- { type: 'stats', ... }      -> { type: 'reset' } | { type: 'off' }
//            <- { type: 'info', speakerF0?, shift?, blockMs?, error? }
//   audio:   -> Float32Array (48 kHz mono, your microphone)  <- Float32Array (48 kHz mono, the AI voice)
const fs = require('node:fs');

const LOG_FILE = process.env.VCAM_ENGINE_LOG || '';
function log(...parts) {
  const line = `[${new Date().toISOString()}] ${parts.join(' ')}\n`;
  try {
    if (LOG_FILE) fs.appendFileSync(LOG_FILE, line);
    else process.stdout.write(line);
  } catch {
    process.stdout.write(line);
  }
}

process.on('uncaughtException', (err) => {
  log(`voice engine crashed: ${(err && err.stack) || err}`);
  process.exit(1);
});

let ort = null;
let settings = null;
const engines = { trained: null, mine: null };
const loads = { trained: null, mine: null };
let active = null; // the engine your microphone goes to
let control = null;
let audio = null;

function reply(message) {
  if (control) control.postMessage(message);
}

function toFloats(data) {
  if (data instanceof Float32Array) return data;
  if (data instanceof ArrayBuffer) return new Float32Array(data);
  if (ArrayBuffer.isView(data)) return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
  return null;
}

// Loads the engine for a kind of voice (once).
function engine(kind) {
  if (engines[kind]) return Promise.resolve(engines[kind]);
  if (!settings) return Promise.reject(new Error('The voice engine is not set up'));
  if (!loads[kind]) {
    loads[kind] = (async () => {
      if (!ort) ort = require('onnxruntime-node');
      let e;
      if (kind === 'trained') {
        const { VoiceConverter } = require('./converter');
        e = new VoiceConverter(ort, settings.modelsDir, { log });
      } else {
        const { MatchConverter } = require('./match');
        e = new MatchConverter(ort, settings.modelsDir, settings.voicesDir, { log });
      }
      e.onAudio = (pcm) => {
        if (audio && active === e) audio.postMessage(pcm);
      };
      e.onInfo = (info) => {
        if (active === e) reply({ type: 'info', ...info });
      };
      const t = Date.now();
      const r = await e.load({ preferred: settings.preferred, gpuCount: settings.gpuCount, onStep: (s) => log(`voice load (${kind}): ${s}`) });
      log(`voice engine (${kind}) loaded on ${r.device} in ${Date.now() - t} ms (${r.ms} ms per block)`);
      e.loadInfo = r;
      engines[kind] = e;
      return e;
    })();
    loads[kind].catch(() => {
      loads[kind] = null;
    });
  }
  return loads[kind];
}

async function handle(msg) {
  try {
    switch (msg.type) {
      case 'load':
        settings = { modelsDir: msg.modelsDir, voicesDir: msg.voicesDir, preferred: msg.preferred || 'auto', gpuCount: msg.gpuCount || 0 };
        reply({ type: 'loaded' });
        break;
      case 'voice': {
        const kind = msg.voice && msg.voice.kind === 'mine' ? 'mine' : 'trained';
        const e = await engine(kind);
        const t = Date.now();
        await e.setVoice(kind === 'mine' ? msg.voice.id : msg.voice);
        if (active !== e) {
          // The previous engine drops its leftover audio, so the new voice starts straight away.
          if (active) active.reset();
          e.reset();
        }
        active = e;
        log(`voice ${kind} ${kind === 'mine' ? msg.voice.id : msg.voice.file} ready in ${Date.now() - t} ms`);
        reply({ type: 'voice', kind, device: e.loadInfo.device, ms: e.loadInfo.ms });
        break;
      }
      case 'prepare': {
        const e = await engine('mine');
        const pcm = toFloats(msg.pcm);
        if (!pcm) throw new Error('The recording could not be read');
        const t = Date.now();
        let last = 0;
        const result = await e.prepare(msg.id, pcm, (progress) => {
          if (progress - last >= 0.04 || progress === 1) {
            last = progress;
            reply({ type: 'prepare-progress', id: msg.id, progress });
          }
        });
        log(`my voice ${msg.id} prepared in ${Date.now() - t} ms: ${result.seconds} s of speech`);
        reply({ type: 'prepared', ...result });
        break;
      }
      case 'convert': {
        const kind = msg.voice && msg.voice.kind === 'mine' ? 'mine' : 'trained';
        const e = await engine(kind);
        const pcm = toFloats(msg.pcm);
        if (!pcm) throw new Error('The recording could not be read');
        let out;
        let rate;
        if (kind === 'mine') {
          const keep = e.voice;
          await e.setVoice(msg.voice.id);
          out = await e.convertWhole(Float32Array.from(pcm));
          rate = 16000;
          if (keep && keep.id !== msg.voice.id) await e.setVoice(keep.id);
        } else {
          await e.setVoice(msg.voice);
          const len = Math.floor(pcm.length / 160) * 160;
          out = await e.convert(Float32Array.from(pcm.subarray(0, len)), len / 160);
          rate = e.voice.sr;
        }
        reply({ type: 'converted', pcm: Float32Array.from(out), rate });
        break;
      }
      case 'stats':
        reply({ type: 'stats', ...(active ? active.stats() : {}) });
        break;
      case 'reset':
        if (active) active.reset();
        break;
      case 'off':
        active = null;
        break;
      default:
        break;
    }
  } catch (err) {
    const message = (err && err.message) || String(err);
    log(`voice ${msg.type} failed: ${message}`);
    reply({ type: 'error', request: msg.type, message });
  }
}

process.parentPort.on('message', (event) => {
  if (!event.data || event.data.type !== 'ports' || !event.ports || event.ports.length < 2) return;
  if (control) control.close();
  if (audio) audio.close();
  [control, audio] = event.ports;
  control.on('message', (e) => handle(e.data));
  let first = true;
  audio.on('message', (e) => {
    const pcm = toFloats(e.data);
    if (first) {
      first = false;
      log(`voice engine: microphone audio arriving (${pcm ? `${pcm.length} samples per chunk` : 'unreadable chunks'})`);
    }
    if (pcm && active && active.ready) active.push(pcm);
  });
  control.start();
  audio.start();
  if (active) active.reset();
  log('voice engine connected to the studio');
  reply({ type: 'hello' });
});

log(`voice engine started (pid ${process.pid})`);
