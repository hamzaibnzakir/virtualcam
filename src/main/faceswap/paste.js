'use strict';

// CPU paste-back of swapped faces, with soft edges. The studio does the same on the GPU
// with a canvas transform; this version is used by tests.
const { invertAffine } = require('./face-math');

// 128x128 alpha mask: solid in the middle, fading to 0 towards the crop edges.
function featherMask(size, inset = 0.1, fade = 0.16) {
  const mask = new Float32Array(size * size);
  const start = size * inset;
  const ramp = size * fade;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const edge = Math.min(x, y, size - 1 - x, size - 1 - y);
      const a = Math.min(1, Math.max(0, (edge - start) / ramp));
      mask[y * size + x] = a * a * (3 - 2 * a); // smoothstep
    }
  }
  return mask;
}

// Blends each face onto the frame using the face's own alpha (the soft edge), bilinear sampled.
function pasteFaces(frame, width, height, faces) {
  for (const face of faces) {
    const size = face.size;
    const toCrop = invertAffine(face.toFrame);
    const corners = [
      [0, 0],
      [size, 0],
      [0, size],
      [size, size],
    ].map(([u, v]) => [
      face.toFrame[0] * u + face.toFrame[1] * v + face.toFrame[2],
      face.toFrame[3] * u + face.toFrame[4] * v + face.toFrame[5],
    ]);
    const minX = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[0]))));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(...corners.map((c) => c[0]))));
    const minY = Math.max(0, Math.floor(Math.min(...corners.map((c) => c[1]))));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(...corners.map((c) => c[1]))));
    for (let y = minY; y <= maxY; y += 1) {
      for (let x = minX; x <= maxX; x += 1) {
        const u = toCrop[0] * x + toCrop[1] * y + toCrop[2];
        const v = toCrop[3] * x + toCrop[4] * y + toCrop[5];
        const u0 = Math.floor(u);
        const v0 = Math.floor(v);
        if (u0 < 0 || v0 < 0 || u0 >= size - 1 || v0 >= size - 1) continue;
        const fu = u - u0;
        const fv = v - v0;
        const px = [0, 0, 0, 0];
        for (let k = 0; k < 4; k += 1) {
          const w = ((k & 1) ? fu : 1 - fu) * ((k >> 1) ? fv : 1 - fv);
          const s = ((v0 + (k >> 1)) * size + u0 + (k & 1)) * 4;
          px[0] += face.rgba[s] * w;
          px[1] += face.rgba[s + 1] * w;
          px[2] += face.rgba[s + 2] * w;
          px[3] += face.rgba[s + 3] * w;
        }
        const a = px[3] / 255;
        if (a <= 0) continue;
        const d = (y * width + x) * 4;
        for (let c = 0; c < 3; c += 1) frame[d + c] = frame[d + c] * (1 - a) + px[c] * a;
      }
    }
  }
  return frame;
}

module.exports = { featherMask, pasteFaces };
