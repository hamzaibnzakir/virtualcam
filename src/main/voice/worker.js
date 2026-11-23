'use strict';

// Voice engine process: the trained AI voices, run away from the app window.
// Two ports from the studio:
//   control: -> { type: 'load', modelsDir, preferred, gpuCount }  <- { type: 'loaded', device, ms } | { type: 'error', request, message }
//            -> { type: 'voice', voice: { file, sr, f0 } }        <- { type: 'voice', file }
//            -> { type: 'stats' }                                 <- { type: 'stats', ... }
//            -> { type: 'reset' }
//            <- { type: 'info', speakerF0?, shift?, blockMs?, error? }
//   audio:   -> Float32Array (48 kHz mono, your microphone)       <- Float32Array (48 kHz mono, the AI voice)
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
let VoiceConverter = null;
let converter = null;
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

async function handle(msg) {
  try {
    switch (msg.type) {
      case 'load': {
        if (!ort) {
          ort = require('onnxruntime-node');
          ({ VoiceConverter } = require('./converter'));
        }
        if (converter) await converter.release();
        converter = new VoiceConverter(ort, msg.modelsDir, { log });
        converter.onAudio = (pcm) => {
          if (audio) audio.postMessage(pcm);
        };
        converter.onInfo = (info) => reply({ type: 'info', ...info });
        const t = Date.now();
        const result = await converter.load({ preferred: msg.preferred, gpuCount: msg.gpuCount, onStep: (s) => log(`voice load: ${s}`) });
        log(`voice engine loaded on ${result.device} in ${Date.now() - t} ms (${result.ms} ms per block)`);
        reply({ type: 'loaded', ...result });
        break;
      }
      case 'voice': {
        if (!converter) throw new Error('The voice engine is not loaded');
        const t = Date.now();
        const result = await converter.setVoice(msg.voice);
        log(`voice ${result.file} ready in ${Date.now() - t} ms`);
        reply({ type: 'voice', ...result });
        break;
      }
      case 'stats':
        reply({ type: 'stats', ...(converter ? converter.stats() : {}) });
        break;
      case 'reset':
        if (converter) converter.reset();
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
    if (pcm && converter && converter.ready) converter.push(pcm);
  });
  control.start();
  audio.start();
  if (converter) converter.reset();
  log('voice engine connected to the studio');
  reply({ type: 'hello' });
});

log(`voice engine started (pid ${process.pid})`);
