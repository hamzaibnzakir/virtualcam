// Draws the program picture (what everyone sees) at a steady frame rate,
// plus the AI input picture when an AI scene is live.
import { drawVideoCover } from './sources.js';
import { OUTPUT_FPS, OUTPUT_HEIGHT, OUTPUT_WIDTH } from './state.js';

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawNameTag(ctx, name, title) {
  if (!name && !title) return;
  const padX = 22;
  const nameFont = '600 30px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
  const titleFont = '400 22px "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';
  ctx.font = nameFont;
  const nameW = name ? ctx.measureText(name).width : 0;
  ctx.font = titleFont;
  const titleW = title ? ctx.measureText(title).width : 0;
  const w = Math.min(OUTPUT_WIDTH * 0.6, Math.max(nameW, titleW) + padX * 2);
  const h = name && title ? 86 : 58;
  const x = 32;
  const y = OUTPUT_HEIGHT - h - 32;

  ctx.fillStyle = 'rgba(11, 12, 16, 0.82)';
  roundRect(ctx, x, y, w, h, 14);
  ctx.fill();

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  if (name) {
    ctx.font = nameFont;
    ctx.fillStyle = '#F2F3F5';
    ctx.fillText(name, x + padX, y + (title ? 38 : 39), w - padX * 2);
  }
  if (title) {
    ctx.font = titleFont;
    ctx.fillStyle = '#B4B9C4';
    ctx.fillText(title, x + padX, y + (name ? 70 : 37), w - padX * 2);
  }
}

export class Compositor {
  constructor(canvas, aiCanvas, sources) {
    this.canvas = canvas;
    this.canvas.width = OUTPUT_WIDTH;
    this.canvas.height = OUTPUT_HEIGHT;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.aiCanvas = aiCanvas;
    this.aiCtx = aiCanvas.getContext('2d', { alpha: false });
    this.sources = sources;

    this.mirror = false;
    this.nameTag = null; // { name, title } or null
    this.aiFeed = false; // keep drawing the AI input picture
    this.aiVideo = null; // transformed video coming back from the AI
    this.aiLive = false; // AI output is trustworthy right now
    this.aiInputStream = null;
    this.aiInputSize = null;

    this.frames = 0;
    this.fps = 0;
    this.lastFpsAt = performance.now();
    this.timer = null;
  }

  configure({ mirror, nameTag }) {
    this.mirror = Boolean(mirror);
    this.nameTag = nameTag && nameTag.enabled ? { name: nameTag.name, title: nameTag.title } : null;
  }

  // The stream fed to the AI. Sized to what the model expects; reused while the size is unchanged.
  aiInput({ width, height, fps }) {
    const same = this.aiInputSize && this.aiInputSize.width === width && this.aiInputSize.height === height;
    if (!same || !this.aiInputStream) {
      if (this.aiInputStream) this.aiInputStream.getTracks().forEach((t) => t.stop());
      this.aiCanvas.width = width;
      this.aiCanvas.height = height;
      this.aiInputSize = { width, height, fps };
      this.drawAiInput();
      this.aiInputStream = this.aiCanvas.captureStream(fps);
    }
    return this.aiInputStream;
  }

  setAiVideo(video) {
    this.aiVideo = video;
  }

  start() {
    if (this.timer) return;
    const interval = 1000 / OUTPUT_FPS;
    let next = performance.now();
    const tick = () => {
      this.render();
      next += interval;
      const now = performance.now();
      if (next < now - interval * 3) next = now; // recover after a long stall
      this.timer = setTimeout(tick, Math.max(0, next - now));
    };
    tick();
  }

  stop() {
    clearTimeout(this.timer);
    this.timer = null;
  }

  drawAiInput() {
    if (!this.aiInputSize) return;
    const { width, height } = this.aiInputSize;
    this.sources.draw(this.aiCtx, width, height, this.mirror);
  }

  render() {
    const { ctx } = this;
    const w = OUTPUT_WIDTH;
    const h = OUTPUT_HEIGHT;

    if (this.aiFeed) this.drawAiInput();

    let drewAi = false;
    if (this.aiLive && this.aiVideo) {
      // The AI already received a mirrored picture when mirroring is on.
      drewAi = drawVideoCover(ctx, this.aiVideo, w, h, false);
    }
    if (!drewAi) this.sources.draw(ctx, w, h, this.mirror);

    if (this.nameTag) drawNameTag(ctx, this.nameTag.name, this.nameTag.title);

    this.frames += 1;
    const now = performance.now();
    if (now - this.lastFpsAt >= 1000) {
      this.fps = Math.round((this.frames * 1000) / (now - this.lastFpsAt));
      this.frames = 0;
      this.lastFpsAt = now;
    }
  }
}
