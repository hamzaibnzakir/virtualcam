// Local face swap adapter. Frames go to the face engine process; swapped faces come back and are
// painted onto that same frame, so the face always lines up with the picture it was made from.
import { output } from './state.js';

const FRESH_MS = 1000; // older results than this are not shown (the real picture shows instead)

export class LocalFaceAI {
  constructor({ api, onState, onError }) {
    this.api = api;
    this.onState = onState;
    this.onError = onError;
    this.state = 'off';
    this.provider = '';
    this.target = null;
    this.dirty = false;
    this.running = false;
    this.engineReady = false;
    this.applied = { refKey: null, enhance: false };
    this.waiters = [];
    this.inFlight = null; // { id, image }
    this.frameId = 0;
    this.lastResultAt = 0;
    this.stats = { ms: 0, fps: 0 };
    this.resultTimes = [];

    // What the engine is fed (mirrored like the output) and the finished picture.
    this.input = document.createElement('canvas');
    this.input.width = output.width;
    this.input.height = output.height;
    this.inputCtx = this.input.getContext('2d', { willReadFrequently: true, alpha: false });
    this.canvas = document.createElement('canvas');
    this.canvas.width = output.width;
    this.canvas.height = output.height;
    this.ctx = this.canvas.getContext('2d', { alpha: false });
    this.ctx.imageSmoothingQuality = 'high';
    this.faceCanvas = document.createElement('canvas');
    this.faceCtx = this.faceCanvas.getContext('2d');

    api.onEngineMessage((msg) => this.handleMessage(msg));
    api.onEngineExit(() => {
      this.engineReady = false;
      this.inFlight = null;
      this.applied = { refKey: null, enhance: false };
      this.rejectWaiters(new Error('The face engine stopped'));
      if (this.target) {
        this.setState('error');
        this.onError('The local face engine stopped. Toggle AI to restart it.');
      }
    });
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState(state);
  }

  get live() {
    return this.state === 'generating' && performance.now() - this.lastResultAt < FRESH_MS;
  }

  // target: null (off) or { refKey, refBlob, enhance }
  setTarget(target) {
    this.target = target;
    this.dirty = true;
    this.pump();
  }

  async pump() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.dirty) {
        this.dirty = false;
        await this.apply(this.target);
      }
    } finally {
      this.running = false;
    }
  }

  async apply(target) {
    if (!target) {
      this.setState('off');
      return;
    }
    try {
      const models = await this.api.modelsStatus();
      if (!models.core) {
        this.setState('needs-models');
        return;
      }
      if (!this.engineReady) await this.startEngine();
      if (this.target !== target) return; // changed while starting; the next pass handles it

      if (target.enhance !== this.applied.enhance) {
        if (target.enhance && !models.boost) {
          this.onError('Download the face detail boost first (Settings in the AI panel).');
        } else {
          this.send({ type: 'enhance', on: target.enhance });
          const res = await this.waitFor('enhance');
          this.applied.enhance = Boolean(res.on);
        }
      }

      const alreadyTried = this.applied.refKey === `missing:${target.refKey}`;
      if (target.refKey !== this.applied.refKey && !alreadyTried) {
        if (!target.refBlob) {
          this.send({ type: 'clear-reference' });
          this.applied.refKey = target.refKey;
        } else {
          this.setState('loading');
          const ref = await blobToPixels(target.refBlob, 1024);
          this.send({ type: 'reference', width: ref.width, height: ref.height, pixels: ref.data });
          const res = await this.waitFor('reference');
          this.applied.refKey = res.found ? target.refKey : `missing:${target.refKey}`;
        }
      }

      if (!target.refBlob) this.setState('needs-reference');
      else if (this.applied.refKey !== target.refKey) this.setState('no-face');
      else this.setState('generating');
    } catch (err) {
      this.setState('error');
      this.onError(`Local AI: ${(err && err.message) || err}`);
    }
  }

  async startEngine() {
    this.setState('starting');
    const res = await this.api.engineStart();
    if (!res || !res.ok) throw new Error('The face engine could not start');
    await this.waitFor('hello', 15000);
    this.send({ type: 'load', modelsDir: res.modelsDir });
    const loaded = await this.waitFor('loaded', 120000);
    this.provider = loaded.provider;
    this.fallbackReason = loaded.fallbackReason || '';
    this.engineReady = true;
    this.applied = { refKey: null, enhance: false };
  }

  send(message) {
    if (!this.api.enginePost(message)) throw new Error('The face engine is not connected');
  }

  waitFor(type, timeout = 60000) {
    return new Promise((resolve, reject) => {
      const waiter = { type, resolve, reject };
      waiter.timer = setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        reject(new Error('The face engine did not answer in time'));
      }, timeout);
      this.waiters.push(waiter);
    });
  }

  rejectWaiters(err) {
    const pending = this.waiters;
    this.waiters = [];
    pending.forEach((w) => {
      clearTimeout(w.timer);
      w.reject(err);
    });
  }

  handleMessage(msg) {
    if (!msg || !msg.type) return;
    if (msg.type === 'result') {
      this.handleResult(msg);
      return;
    }
    if (msg.type === 'error') {
      if (msg.request === 'frame') {
        this.inFlight = null;
        return;
      }
      const waiter = this.waiters.find((w) => w.type === msg.request || (msg.request === 'load' && w.type === 'loaded'));
      if (waiter) {
        this.waiters = this.waiters.filter((w) => w !== waiter);
        clearTimeout(waiter.timer);
        waiter.reject(new Error(msg.message));
      } else {
        this.onError(`Local AI: ${msg.message}`);
      }
      return;
    }
    const waiter = this.waiters.find((w) => w.type === msg.type);
    if (waiter) {
      this.waiters = this.waiters.filter((w) => w !== waiter);
      clearTimeout(waiter.timer);
      waiter.resolve(msg);
    }
  }

  // Called by the compositor every frame with a function that draws the current source.
  tick(drawSource) {
    if (this.state !== 'generating' || !this.engineReady || this.inFlight) return;
    if (this.input.width !== output.width || this.input.height !== output.height) {
      // Picture shape changed: follow it (canvas contexts keep their settings).
      this.input.width = output.width;
      this.input.height = output.height;
      this.canvas.width = output.width;
      this.canvas.height = output.height;
      this.ctx.imageSmoothingQuality = 'high';
    }
    drawSource(this.inputCtx, output.width, output.height);
    const image = this.inputCtx.getImageData(0, 0, output.width, output.height);
    const id = ++this.frameId;
    this.inFlight = { id, image, sentAt: performance.now() };
    try {
      this.send({ type: 'frame', id, width: image.width, height: image.height, pixels: image.data });
    } catch {
      this.inFlight = null;
    }
  }

  handleResult(msg) {
    const pending = this.inFlight;
    if (!pending || pending.id !== msg.id) return;
    this.inFlight = null;
    if (msg.skipped) return;

    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.putImageData(pending.image, 0, 0);
    for (const face of msg.faces || []) {
      const size = face.size;
      if (this.faceCanvas.width !== size) {
        this.faceCanvas.width = size;
        this.faceCanvas.height = size;
      }
      this.faceCtx.putImageData(new ImageData(new Uint8ClampedArray(face.pixels), size, size), 0, 0);
      const m = face.toFrame; // [a, c, e, b, d, f]
      this.ctx.setTransform(m[0], m[3], m[1], m[4], m[2], m[5]);
      this.ctx.drawImage(this.faceCanvas, 0, 0);
    }
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);

    const now = performance.now();
    this.lastResultAt = now;
    this.stats.ms = msg.ms || Math.round(now - pending.sentAt);
    this.resultTimes.push(now);
    while (this.resultTimes.length && now - this.resultTimes[0] > 1000) this.resultTimes.shift();
    this.stats.fps = this.resultTimes.length;
  }
}

async function blobToPixels(blob, maxSide) {
  const bitmap = await createImageBitmap(blob);
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return ctx.getImageData(0, 0, width, height);
}
