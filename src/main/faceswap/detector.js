'use strict';

// SCRFD face detector (insightface det_10g): finds faces and their 5 landmarks.
const { letterboxTensor } = require('./face-math');

const STRIDES = [8, 16, 32];
const ANCHORS = 2;

function iou(a, b) {
  const x1 = Math.max(a.box[0], b.box[0]);
  const y1 = Math.max(a.box[1], b.box[1]);
  const x2 = Math.min(a.box[2], b.box[2]);
  const y2 = Math.min(a.box[3], b.box[3]);
  const inter = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
  const areaA = (a.box[2] - a.box[0] + 1) * (a.box[3] - a.box[1] + 1);
  const areaB = (b.box[2] - b.box[0] + 1) * (b.box[3] - b.box[1] + 1);
  return inter / (areaA + areaB - inter);
}

class FaceDetector {
  constructor(ort, session, { size = 640, threshold = 0.5, nms = 0.4 } = {}) {
    this.ort = ort;
    this.session = session;
    this.size = size;
    this.threshold = threshold;
    this.nms = nms;
    this.buffer = new Float32Array(3 * size * size);
  }

  // Returns faces sorted by size (largest first): { box: [x1,y1,x2,y2], kps: [[x,y]x5], score }
  async detect(rgba, width, height) {
    const { size } = this;
    const { tensor, scale } = letterboxTensor(rgba, width, height, size, 127.5, 128, this.buffer);
    const input = new this.ort.Tensor('float32', tensor, [1, 3, size, size]);
    const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
    const out = this.session.outputNames.map((name) => outputs[name].data);

    const faces = [];
    STRIDES.forEach((stride, level) => {
      const scores = out[level];
      const boxes = out[level + 3];
      const kps = out[level + 6];
      const gridW = Math.floor(size / stride);
      for (let i = 0; i < scores.length; i += 1) {
        const score = scores[i];
        if (score < this.threshold) continue;
        const cell = Math.floor(i / ANCHORS);
        const cx = (cell % gridW) * stride;
        const cy = Math.floor(cell / gridW) * stride;
        const b = i * 4;
        const k = i * 10;
        const box = [
          (cx - boxes[b] * stride) / scale,
          (cy - boxes[b + 1] * stride) / scale,
          (cx + boxes[b + 2] * stride) / scale,
          (cy + boxes[b + 3] * stride) / scale,
        ];
        const points = [];
        for (let p = 0; p < 5; p += 1) {
          points.push([(cx + kps[k + p * 2] * stride) / scale, (cy + kps[k + p * 2 + 1] * stride) / scale]);
        }
        faces.push({ box, kps: points, score });
      }
    });

    faces.sort((a, b) => b.score - a.score);
    const kept = [];
    for (const face of faces) {
      if (kept.every((k) => iou(k, face) <= this.nms)) kept.push(face);
    }
    const area = (f) => (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]);
    return kept.sort((a, b) => area(b) - area(a));
  }
}

module.exports = { FaceDetector };
