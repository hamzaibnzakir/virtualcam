'use strict';

// Image helpers for the face pipeline. Images are RGBA Uint8Arrays (width * height * 4).

// ArcFace's canonical 5-point layout (eyes, nose, mouth corners) in a 112x112 crop.
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

// Where the 5 points should land in a square crop of `size` pixels (matches insightface).
function alignmentTarget(size) {
  let ratio;
  let diffX;
  if (size % 112 === 0) {
    ratio = size / 112;
    diffX = 0;
  } else {
    ratio = size / 128;
    diffX = 8 * ratio;
  }
  return ARCFACE_DST.map(([x, y]) => [x * ratio + diffX, y * ratio]);
}

// Least-squares similarity transform (rotation + uniform scale + translation)
// mapping src points onto dst points. Returns a 2x3 matrix [a, c, e, b, d, f]
// meaning x' = a*x + c*y + e, y' = b*x + d*y + f.
function similarity(src, dst) {
  const n = src.length;
  let msx = 0;
  let msy = 0;
  let mdx = 0;
  let mdy = 0;
  for (let i = 0; i < n; i += 1) {
    msx += src[i][0];
    msy += src[i][1];
    mdx += dst[i][0];
    mdy += dst[i][1];
  }
  msx /= n;
  msy /= n;
  mdx /= n;
  mdy /= n;
  let numA = 0;
  let numB = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    const sx = src[i][0] - msx;
    const sy = src[i][1] - msy;
    const dx = dst[i][0] - mdx;
    const dy = dst[i][1] - mdy;
    numA += sx * dx + sy * dy;
    numB += sx * dy - sy * dx;
    den += sx * sx + sy * sy;
  }
  if (den === 0) throw new Error('Degenerate landmarks');
  const a = numA / den;
  const b = numB / den;
  const e = mdx - (a * msx - b * msy);
  const f = mdy - (b * msx + a * msy);
  return [a, -b, e, b, a, f];
}

function invertAffine(m) {
  const [a, c, e, b, d, f] = m;
  const det = a * d - b * c;
  if (det === 0) throw new Error('Non-invertible transform');
  const ia = d / det;
  const ic = -c / det;
  const ib = -b / det;
  const id = a / det;
  return [ia, ic, -(ia * e + ic * f), ib, id, -(ib * e + id * f)];
}

// Samples src through matrix m (src -> dst) into a size x size crop.
// Writes a planar float tensor [3, size, size] with value = (px - mean) / std.
function warpToTensor(src, srcW, srcH, m, size, mean, std, out = new Float32Array(3 * size * size)) {
  const inv = invertAffine(m);
  const plane = size * size;
  for (let v = 0; v < size; v += 1) {
    for (let u = 0; u < size; u += 1) {
      const x = inv[0] * u + inv[1] * v + inv[2];
      const y = inv[3] * u + inv[4] * v + inv[5];
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;
      let r = 0;
      let g = 0;
      let bl = 0;
      // Bilinear sample; pixels outside the image count as black (like OpenCV's default border).
      for (let k = 0; k < 4; k += 1) {
        const xx = x0 + (k & 1);
        const yy = y0 + (k >> 1);
        if (xx < 0 || yy < 0 || xx >= srcW || yy >= srcH) continue;
        const w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
        const idx = (yy * srcW + xx) * 4;
        r += src[idx] * w;
        g += src[idx + 1] * w;
        bl += src[idx + 2] * w;
      }
      const o = v * size + u;
      out[o] = (r - mean) / std;
      out[plane + o] = (g - mean) / std;
      out[2 * plane + o] = (bl - mean) / std;
    }
  }
  return out;
}

// Resizes src into the top-left of a square `size` canvas (rest black), the way SCRFD expects.
// Returns the planar tensor and the scale applied.
function letterboxTensor(src, srcW, srcH, size, mean, std, out = new Float32Array(3 * size * size)) {
  const scale = srcH / srcW > 1 ? size / srcH : size / srcW;
  const newW = Math.max(1, Math.floor(srcW * scale));
  const newH = Math.max(1, Math.floor(srcH * scale));
  const plane = size * size;
  const pad = (0 - mean) / std;
  out.fill(pad);
  const sx = srcW / newW;
  const sy = srcH / newH;
  for (let y = 0; y < newH; y += 1) {
    const fy = (y + 0.5) * sy - 0.5;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = Math.min(1, Math.max(0, fy - y0));
    for (let x = 0; x < newW; x += 1) {
      const fx = (x + 0.5) * sx - 0.5;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const wx = Math.min(1, Math.max(0, fx - x0));
      const i00 = (y0 * srcW + x0) * 4;
      const i01 = (y0 * srcW + x1) * 4;
      const i10 = (y1 * srcW + x0) * 4;
      const i11 = (y1 * srcW + x1) * 4;
      const o = y * size + x;
      for (let c = 0; c < 3; c += 1) {
        const top = src[i00 + c] * (1 - wx) + src[i01 + c] * wx;
        const bottom = src[i10 + c] * (1 - wx) + src[i11 + c] * wx;
        out[c * plane + o] = (top * (1 - wy) + bottom * wy - mean) / std;
      }
    }
  }
  return { tensor: out, scale: newH / srcH };
}

// compose(A, B) applies B first, then A.
function compose(a, b) {
  return [
    a[0] * b[0] + a[1] * b[3],
    a[0] * b[1] + a[1] * b[4],
    a[0] * b[2] + a[1] * b[5] + a[2],
    a[3] * b[0] + a[4] * b[3],
    a[3] * b[1] + a[4] * b[4],
    a[3] * b[2] + a[4] * b[5] + a[5],
  ];
}

// Samples an RGBA image through m (src -> dst) into an RGBA crop of size x size (bilinear).
function warpRgba(src, srcW, srcH, m, size) {
  const inv = invertAffine(m);
  const out = new Uint8ClampedArray(size * size * 4);
  for (let v = 0; v < size; v += 1) {
    for (let u = 0; u < size; u += 1) {
      const x = inv[0] * u + inv[1] * v + inv[2];
      const y = inv[3] * u + inv[4] * v + inv[5];
      const x0 = Math.floor(x);
      const y0 = Math.floor(y);
      const fx = x - x0;
      const fy = y - y0;
      const acc = [0, 0, 0, 0];
      for (let k = 0; k < 4; k += 1) {
        const xx = x0 + (k & 1);
        const yy = y0 + (k >> 1);
        if (xx < 0 || yy < 0 || xx >= srcW || yy >= srcH) continue;
        const w = ((k & 1) ? fx : 1 - fx) * ((k >> 1) ? fy : 1 - fy);
        const idx = (yy * srcW + xx) * 4;
        acc[0] += src[idx] * w;
        acc[1] += src[idx + 1] * w;
        acc[2] += src[idx + 2] * w;
        acc[3] += src[idx + 3] * w;
      }
      const o = (v * size + u) * 4;
      out[o] = acc[0];
      out[o + 1] = acc[1];
      out[o + 2] = acc[2];
      out[o + 3] = acc[3];
    }
  }
  return out;
}

function l2normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i += 1) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum) || 1;
  const out = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i += 1) out[i] = vec[i] / norm;
  return out;
}

module.exports = { alignmentTarget, similarity, invertAffine, compose, warpToTensor, warpRgba, letterboxTensor, l2normalize };
