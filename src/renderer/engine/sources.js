// Loads and draws scene sources: webcam, image, video loop, screen and text cards.

function drawCover(ctx, el, srcW, srcH, w, h, mirror) {
  if (!srcW || !srcH) return false;
  const scale = Math.max(w / srcW, h / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const dx = (w - dw) / 2;
  const dy = (h - dh) / 2;
  if (mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(el, w - dx - dw, dy, dw, dh);
    ctx.restore();
  } else {
    ctx.drawImage(el, dx, dy, dw, dh);
  }
  return true;
}

// Whole picture visible, centred; the leftover area gets a soft, darkened, blown-up copy.
const blurCanvas = document.createElement('canvas');
const blurCtx = blurCanvas.getContext('2d');
function drawFit(ctx, el, srcW, srcH, w, h, mirror) {
  if (!srcW || !srcH) return false;
  blurCanvas.width = 64;
  blurCanvas.height = Math.max(1, Math.round((64 * h) / w));
  // Blurring a tiny copy is cheap; scaling it up smooths it further.
  blurCtx.filter = 'blur(3px)';
  drawCover(blurCtx, el, srcW, srcH, blurCanvas.width, blurCanvas.height, mirror);
  blurCtx.filter = 'none';
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(blurCanvas, 0, 0, w, h);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
  ctx.fillRect(0, 0, w, h);
  const scale = Math.min(w / srcW, h / srcH);
  const dw = srcW * scale;
  const dh = srcH * scale;
  const dx = (w - dw) / 2;
  const dy = (h - dh) / 2;
  if (mirror) {
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(el, w - dx - dw, dy, dw, dh);
    ctx.restore();
  } else {
    ctx.drawImage(el, dx, dy, dw, dh);
  }
  return true;
}

function drawSized(ctx, el, srcW, srcH, w, h, mirror, fit) {
  return fit === 'fit' ? drawFit(ctx, el, srcW, srcH, w, h, mirror) : drawCover(ctx, el, srcW, srcH, w, h, mirror);
}

export function drawVideoCover(ctx, video, w, h, mirror = false) {
  if (!video || video.readyState < 2) return false;
  return drawCover(ctx, video, video.videoWidth, video.videoHeight, w, h, mirror);
}

// Video, canvas or image: fills w x h, cropping the overflow.
export function drawAnyCover(ctx, el, w, h, mirror = false) {
  if (!el) return false;
  if (el instanceof HTMLVideoElement) return drawVideoCover(ctx, el, w, h, mirror);
  if (el instanceof HTMLCanvasElement) return drawCover(ctx, el, el.width, el.height, w, h, mirror);
  return drawCover(ctx, el, el.naturalWidth, el.naturalHeight, w, h, mirror);
}

function wrapLines(ctx, text, maxWidth) {
  const words = text.split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = word;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines.slice(0, 4);
}

export function drawCard(ctx, w, h, text, color) {
  ctx.fillStyle = color || '#1B1E27';
  ctx.fillRect(0, 0, w, h);
  const size = Math.round(h * 0.11);
  ctx.fillStyle = '#F2F3F5';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `700 ${size}px "Segoe UI Variable Display", "Segoe UI", system-ui, sans-serif`;
  const lines = wrapLines(ctx, text || ' ', w * 0.8);
  const lineH = size * 1.15;
  const top = h / 2 - ((lines.length - 1) * lineH) / 2;
  lines.forEach((line, i) => ctx.fillText(line, w / 2, top + i * lineH));
}

export function drawPlaceholder(ctx, w, h, message) {
  ctx.fillStyle = '#101218';
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#9BA1AE';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `500 ${Math.round(h * 0.035)}px "Segoe UI", system-ui, sans-serif`;
  ctx.fillText(message, w / 2, h / 2);
}

function friendlyMediaError(err, what) {
  const name = err && err.name;
  if (name === 'NotAllowedError') return `${what} access was blocked. Check your privacy settings.`;
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return `No ${what.toLowerCase()} found.`;
  if (name === 'NotReadableError') return `${what} is busy in another app.`;
  return `${what} could not start: ${(err && err.message) || 'unknown error'}`;
}

function sourceKey(spec) {
  if (spec.type === 'webcam') return `webcam:${spec.deviceId || 'default'}`;
  if (spec.type === 'image' || spec.type === 'video') return `${spec.type}:${spec.filePath || ''}`;
  return spec.type;
}

export class SourceManager {
  constructor(stage, { readFile, onStatus }) {
    this.stage = stage; // hidden container so media elements keep decoding
    this.readFile = readFile;
    this.onStatus = onStatus || (() => {});
    this.spec = null;
    this.token = 0;
    this.current = null; // { kind, el } or { kind: 'card' }
    this.status = { ready: false, message: 'Starting…' };
    this.webcams = new Map(); // deviceKey -> { stream, video }
    this.screen = null; // { stream, video }
    this.files = new Map(); // path -> { kind, url, el }
  }

  setStatus(ready, message = '') {
    this.status = { ready, message };
    this.onStatus(this.status);
  }

  // Switches to a new source. The previous picture keeps showing until the new one is ready,
  // so switching scenes never flashes black.
  async use(spec) {
    const token = ++this.token;
    const key = sourceKey(spec);
    this.pauseVideosExcept(spec.type === 'video' ? spec.filePath : null);

    if (spec.type === 'card') {
      this.switchTo(spec, key, { kind: 'card' });
      this.setStatus(true);
      return;
    }

    if (key === this.currentKey && this.current) {
      this.spec = { ...spec };
      if (this.current.kind === 'video') this.current.el.play().catch(() => {});
      this.setStatus(true);
      return;
    }

    if ((spec.type === 'image' || spec.type === 'video') && !spec.filePath) {
      this.switchTo(spec, key, null);
      this.setStatus(false, `Choose ${spec.type === 'image' ? 'an image' : 'a video'} for this scene`);
      return;
    }

    this.setStatus(false, 'Loading source…');
    try {
      let el;
      if (spec.type === 'webcam') el = await this.webcam(spec.deviceId);
      else if (spec.type === 'screen') el = await this.screenVideo();
      else el = await this.fileElement(spec.type, spec.filePath);
      if (token !== this.token) return; // a newer scene was picked meanwhile
      this.switchTo(spec, key, { kind: spec.type, el });
      if (spec.type === 'video') el.play().catch(() => {});
      this.setStatus(true);
    } catch (err) {
      if (token !== this.token) return;
      const what = { webcam: 'Camera', screen: 'Screen capture', image: 'Image', video: 'Video' }[spec.type];
      this.switchTo(spec, key, null);
      this.setStatus(false, friendlyMediaError(err, what));
    }
  }

  switchTo(spec, key, current) {
    this.spec = { ...spec };
    this.currentKey = key;
    this.current = current;
  }

  async webcam(deviceId) {
    const key = deviceId || 'default';
    const cached = this.webcams.get(key);
    if (cached && cached.stream.getVideoTracks().some((t) => t.readyState === 'live')) return cached.video;

    const video = { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } };
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: deviceId ? { ...video, deviceId: { exact: deviceId } } : video,
      });
    } catch (err) {
      if (!deviceId) throw err;
      // The saved camera is gone (unplugged): fall back to the default one.
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video });
    }
    const el = this.makeVideo();
    el.srcObject = stream;
    await el.play().catch(() => {});
    this.webcams.set(key, { stream, video: el });
    return el;
  }

  async screenVideo() {
    if (this.screen && this.screen.stream.getVideoTracks().some((t) => t.readyState === 'live')) return this.screen.video;
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false });
    const el = this.makeVideo();
    el.srcObject = stream;
    await el.play().catch(() => {});
    this.screen = { stream, video: el };
    stream.getVideoTracks().forEach((t) =>
      t.addEventListener('ended', () => {
        this.screen = null;
        if (this.spec && this.spec.type === 'screen') this.setStatus(false, 'Screen capture stopped');
      })
    );
    return el;
  }

  async fileElement(kind, filePath) {
    const cached = this.files.get(filePath);
    if (cached && cached.kind === kind) return cached.el;

    const bytes = await this.readFile(filePath);
    const url = URL.createObjectURL(new Blob([bytes]));
    let el;
    try {
      if (kind === 'image') {
        el = new Image();
        el.src = url;
        await el.decode();
      } else {
        el = this.makeVideo();
        el.loop = true;
        el.src = url;
        await new Promise((resolve, reject) => {
          el.addEventListener('loadeddata', resolve, { once: true });
          el.addEventListener('error', () => reject(new Error('This video format is not supported')), { once: true });
        });
      }
    } catch (err) {
      URL.revokeObjectURL(url);
      throw err;
    }
    this.files.set(filePath, { kind, url, el });
    this.trimFileCache(filePath);
    return el;
  }

  trimFileCache(keepPath) {
    // Keep a handful of loaded files around for instant switching; release the rest.
    const entries = [...this.files.entries()];
    while (entries.length > 6) {
      const [path, entry] = entries.shift();
      if (path === keepPath) continue;
      if (entry.kind === 'video') {
        entry.el.pause();
        entry.el.removeAttribute('src');
        entry.el.load();
        entry.el.remove();
      }
      URL.revokeObjectURL(entry.url);
      this.files.delete(path);
    }
  }

  forgetFile(filePath) {
    const entry = this.files.get(filePath);
    if (!entry) return;
    if (entry.kind === 'video') {
      entry.el.pause();
      entry.el.remove();
    }
    URL.revokeObjectURL(entry.url);
    this.files.delete(filePath);
  }

  pauseVideosExcept(filePath) {
    for (const [path, entry] of this.files) {
      if (entry.kind === 'video' && path !== filePath) entry.el.pause();
    }
  }

  makeVideo() {
    const el = document.createElement('video');
    el.muted = true;
    el.playsInline = true;
    this.stage.appendChild(el);
    return el;
  }

  // Draws the current source into ctx ('fill' crops to fill, 'fit' shows it all). True when real content was drawn.
  draw(ctx, w, h, mirror, fit = 'fill') {
    const spec = this.spec;
    const cur = this.current;
    if (spec && cur && cur.kind === 'card') {
      drawCard(ctx, w, h, spec.text, spec.color);
      return true;
    }
    if (!cur || !cur.el) {
      drawPlaceholder(ctx, w, h, this.status.message || 'Loading source…');
      return false;
    }
    let ok = false;
    if (cur.kind === 'image') {
      ok = drawSized(ctx, cur.el, cur.el.naturalWidth, cur.el.naturalHeight, w, h, mirror, fit);
    } else if (cur.el.readyState >= 2) {
      ok = drawSized(ctx, cur.el, cur.el.videoWidth, cur.el.videoHeight, w, h, mirror, fit);
    }
    if (!ok) drawPlaceholder(ctx, w, h, 'Waiting for frames…');
    return ok;
  }

  async listCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices
        .filter((d) => d.kind === 'videoinput')
        .map((d, i) => ({ id: d.deviceId, label: d.label || `Camera ${i + 1}` }));
    } catch {
      return [];
    }
  }
}
