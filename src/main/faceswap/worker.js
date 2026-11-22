'use strict';

// Face engine process. Runs the local face swap away from the app window so the UI never stutters.
// The studio talks to it directly over a MessagePort:
//   -> { type: 'load', modelsDir, preferred }        <- { type: 'loaded', provider } | { type: 'error', ... }
//   -> { type: 'reference', width, height, pixels }  <- { type: 'reference', found }
//   -> { type: 'clear-reference' }
//   -> { type: 'enhance', on }                       <- { type: 'enhance', on }
//   -> { type: 'frame', id, width, height, pixels }  <- { type: 'result', id, faces, ms }
const ort = require('onnxruntime-node');
const { FaceSwapper } = require('./swapper');

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

async function handle(msg) {
  try {
    switch (msg.type) {
      case 'load': {
        swapper = new FaceSwapper(ort, msg.modelsDir);
        const { provider, fallbackReason } = await swapper.load({ preferred: msg.preferred || 'auto' });
        reply({ type: 'loaded', provider, fallbackReason });
        break;
      }
      case 'reference': {
        if (!swapper) throw new Error('Face engine is not loaded');
        const found = await swapper.setReference(toBytes(msg.pixels), msg.width, msg.height);
        reply({ type: 'reference', found });
        break;
      }
      case 'clear-reference':
        if (swapper) swapper.clearReference();
        break;
      case 'enhance': {
        if (!swapper) throw new Error('Face engine is not loaded');
        const on = await swapper.setEnhance(Boolean(msg.on));
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
      default:
        break;
    }
  } catch (err) {
    reply({ type: 'error', request: msg.type, id: msg.id, message: (err && err.message) || String(err) });
  }
}

process.parentPort.on('message', (event) => {
  // The app hands over the port that connects us to the studio window.
  if (event.data && event.data.type === 'port' && event.ports && event.ports[0]) {
    port = event.ports[0];
    port.on('message', (e) => handle(e.data));
    port.start();
    reply({ type: 'hello' });
  }
});
