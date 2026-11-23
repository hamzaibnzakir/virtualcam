'use strict';

// Face engine process. Runs the local face swap away from the app window so the UI never stutters.
// The studio talks to it directly over a MessagePort:
//   -> { type: 'load', modelsDir, preferred }        <- { type: 'loaded', provider, deviceLabel, ... } | { type: 'error', ... }
//   -> { type: 'swap', id, pixels }  (128x128 aligned crop)  <- { type: 'swapped', id, pixels, size, toCrop, ms, stages } | { skipped }
//   While a request is running:                      <- { type: 'progress', request, step, seconds } every few seconds
//   -> { type: 'reference', width, height, pixels }  <- { type: 'reference', found }
//   -> { type: 'clear-reference' }
//   -> { type: 'enhance', on }                       <- { type: 'enhance', on }
//   -> { type: 'frame', id, width, height, pixels }  <- { type: 'result', id, faces, ms }
const fs = require('node:fs');

// Everything written here goes to face-engine.log in the app's logs folder (written straight away,
// so nothing is lost if the process dies).
const LOG_FILE = process.env.VCAM_ENGINE_LOG || '';
function log(...parts) {
  const line = `[${new Date().toISOString()}] ${parts.join(' ')}\n`;
  if (LOG_FILE) {
    try {
      fs.appendFileSync(LOG_FILE, line);
      return;
    } catch {
      // fall through to stdout
    }
  }
  process.stdout.write(line);
}

process.on('uncaughtException', (err) => {
  log(`crashed: ${(err && err.stack) || err}`);
  process.exit(1);
});

// The AI runtime is loaded on the first 'load' request, not at start-up, so the studio hears from
// this process straight away and a slow or failing runtime load is reported like any other step.
let ort = null;
let FaceSwapper = null;
function loadRuntime() {
  if (ort) return;
  const t = Date.now();
  try {
    ort = require('onnxruntime-node');
    ({ FaceSwapper } = require('./swapper'));
  } catch (err) {
    ort = null;
    throw new Error(`The AI runtime could not load: ${(err && err.message) || err}`);
  }
  log(`AI runtime loaded in ${Date.now() - t} ms (onnxruntime ${ort.env && ort.env.versions ? ort.env.versions.common : ''})`);
}
log(`face engine started (pid ${process.pid})`);

let swapper = null;
let port = null;
let busy = false;

function reply(message) {
  if (port) port.postMessage(message);
}

function toBytes(pixels) {
  if (pixels instanceof Uint8Array) return pixels;
  if (pixels instanceof ArrayBuffer) return new Uint8Array(pixels);
  if (ArrayBuffer.isView(pixels)) return new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  throw new Error('Frame pixels are missing');
}

const LONG_REQUESTS = new Set(['load', 'reference', 'enhance']);

async function handle(msg) {
  // Long requests report what they are doing every few seconds, so the studio waits as long as work
  // is still going on and can say which step is slow.
  let step = msg.type === 'load' ? 'starting' : msg.type;
  const started = Date.now();
  const heartbeat = LONG_REQUESTS.has(msg.type)
    ? setInterval(() => reply({ type: 'progress', request: msg.type, step, seconds: Math.round((Date.now() - started) / 1000) }), 3000)
    : null;
  const onStep = (next) => {
    step = next;
    log(`${msg.type}: ${next}`);
    reply({ type: 'progress', request: msg.type, step, seconds: Math.round((Date.now() - started) / 1000) });
  };
  try {
    switch (msg.type) {
      case 'load': {
        onStep('loading the AI runtime');
        loadRuntime();
        swapper = new FaceSwapper(ort, msg.modelsDir);
        const info = await swapper.load({ preferred: msg.preferred || 'auto', gpuCount: msg.gpuCount || 0, swapFile: msg.swapFile || null, onStep, log });
        log(`load: done in ${Date.now() - started} ms on ${info.deviceLabel} (${info.swapMs} ms per swap)`);
        reply({ type: 'loaded', ...info });
        break;
      }
      case 'reference': {
        if (!swapper) throw new Error('Face engine is not loaded');
        onStep('reading the face in the photo');
        const found = await swapper.setReference(toBytes(msg.pixels), msg.width, msg.height);
        log(`reference: face ${found ? 'found' : 'not found'} in ${Date.now() - started} ms`);
        reply({ type: 'reference', found });
        break;
      }
      case 'clear-reference':
        if (swapper) swapper.clearReference();
        break;
      case 'enhance': {
        if (!swapper) throw new Error('Face engine is not loaded');
        if (msg.on) onStep('loading the face detail model');
        const on = await swapper.setEnhance(Boolean(msg.on));
        log(`enhance: ${on ? 'on' : 'off'} after ${Date.now() - started} ms`);
        reply({ type: 'enhance', on });
        break;
      }
      case 'frame': {
        // One frame at a time: if the studio sends faster than we can swap, extra frames are skipped.
        if (!swapper || busy) {
          reply({ type: 'result', id: msg.id, skipped: true, faces: [] });
          break;
        }
        busy = true;
        const started = Date.now();
        try {
          const faces = await swapper.process(toBytes(msg.pixels), msg.width, msg.height);
          reply({
            type: 'result',
            id: msg.id,
            ms: Date.now() - started,
            faces: faces.map((f) => ({ size: f.size, toFrame: f.toFrame, pixels: f.rgba })),
          });
        } finally {
          busy = false;
        }
        break;
      }
      case 'swap': {
        // A face the studio already tracked, cropped and aligned (128x128): no face finding here.
        if (!swapper || busy) {
          reply({ type: 'swapped', id: msg.id, skipped: true });
          break;
        }
        busy = true;
        const started = Date.now();
        try {
          const face = await swapper.swapCrop(toBytes(msg.pixels));
          if (!face) reply({ type: 'swapped', id: msg.id, skipped: true });
          else reply({ type: 'swapped', id: msg.id, ms: Date.now() - started, stages: face.ms, size: face.size, toCrop: face.toCrop, pixels: face.rgba });
        } finally {
          busy = false;
        }
        break;
      }
      default:
        break;
    }
  } catch (err) {
    const message = (err && err.message) || String(err);
    if (msg.type !== 'frame' && msg.type !== 'swap') log(`${msg.type}: failed while ${step}: ${message}`);
    reply({ type: 'error', request: msg.type, id: msg.id, step, message });
  } finally {
    if (heartbeat) clearInterval(heartbeat);
  }
}

process.parentPort.on('message', (event) => {
  // The app hands over the port that connects us to the studio window.
  if (event.data && event.data.type === 'port' && event.ports && event.ports[0]) {
    port = event.ports[0];
    port.on('message', (e) => handle(e.data));
    port.start();
    log('connected to the studio');
    reply({ type: 'hello' });
  }
});
