// Lite engine: "wear" the face from a photo with no generative AI, so it runs on almost any PC.
//
//   1. Track 478 face points with MediaPipe (small, runs on the CPU). The face's shape is smoothed in
//      the head's own frame, so the head's moves aren't delayed; brief tracking losses fade, not pop.
//   2. Warp the photo's face onto yours with a fixed triangle mesh built once from the photo,
//      leaving holes for your real eyes and mouth so blinks, speech and teeth stay yours.
//   3. In a WebGL shader: match the photo's colours to your camera in LAB space, take the room's
//      shading from your camera (frequency separation), add back your skin highlights, and feather
//      the edges. Triangles that turn away from the camera fade out instead of smearing. If you
//      wear glasses, their rims stay on top of the photo's face.
import { FaceLandmarker } from '@mediapipe/tasks-vision';
import Delaunator from 'delaunator';
import { output } from './state.js';

const FRESH_MS = 1000;
// Tunable look of the Lite engine (the benchmark adjusts these; the app uses the defaults).
//   shading:  how much of the camera's broad lighting is laid over the photo's face
//   detail:   how much of the camera's fine skin detail/highlights is added back
//   blurDiv:  lighting blur radius = face width / blurDiv
//   contrast: 0 = take the camera's colour contrast, 1 = keep the photo's
//   feather:  soft edge at the face outline, as a share of face width
//   holeFeather: soft edge around the eye holes, as a share of face width
//   mouthFeather: soft edge around the mouth hole, as a share of face width
//   eyeEdge:  photo opacity right at the eye opening (0 = fade to your eye, 1 = hard edge)
//   glassesCam: 1 = keep the rims of your own glasses on top of the photo's face (when detected)
//   spanCut:  1 = also cut triangles that join the upper and lower lid/lip (closed eyes/mouth in the photo)
//   minCutoff, beta, dCutoff: one-euro smoothing of the face's shape in the head's own frame (Hz when
//             still; extra Hz per eye distance/s of shape change; Hz for the speed estimate)
//   turnLo, turnHi: a triangle fades out as its size falls from turnHi to turnLo of the photo's
//                   (it is turning away from the camera)
//   holdMs:   when tracking drops out, keep the last points this long while the face fades out
//   fadeInMs, fadeOutMs: how long the face takes to fade in when found and out when lost
export const liteTuning = {
  shading: 0.15,
  detail: 0.03,
  blurDiv: 7,
  contrast: 0.5,
  feather: 0.03,
  holeFeather: 0.01,
  mouthFeather: 0.01,
  eyeEdge: 0.75,
  spanCut: 1,
  glassesCam: 1,
  minCutoff: 2,
  beta: 12,
  dCutoff: 1,
  turnLo: 0.05,
  turnHi: 0.35,
  holdMs: 250,
  fadeInMs: 150,
  fadeOutMs: 200,
};
const TUNE = liteTuning;
const MESH_POINTS = 468; // 0..467 are the face; 468..477 are the irises
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const LEFT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
const INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const HOLES = [RIGHT_EYE, LEFT_EYE, INNER_LIPS];
// Upper and lower lids/lips (two rings each, corners left out), per hole.
const HOLE_SIDES = [
  [[246, 161, 160, 159, 158, 157, 173, 247, 30, 29, 27, 28, 56, 190], [7, 163, 144, 145, 153, 154, 155, 25, 110, 24, 23, 22, 26, 112]],
  [[466, 388, 387, 386, 385, 384, 398, 467, 260, 259, 257, 258, 286, 414], [249, 390, 373, 374, 380, 381, 382, 255, 339, 254, 253, 252, 256, 341]],
  [
    [185, 40, 39, 37, 0, 267, 269, 270, 409, 184, 74, 73, 72, 11, 302, 303, 304, 408, 183, 42, 41, 38, 12, 268, 271, 272, 407, 191, 80, 81, 82, 13, 312, 311, 310, 415],
    [146, 91, 181, 84, 17, 314, 405, 321, 375, 77, 90, 180, 85, 16, 315, 404, 320, 307, 96, 89, 179, 86, 15, 316, 403, 319, 325, 95, 88, 178, 87, 14, 317, 402, 318, 324],
  ],
];

// ---------- small geometry helpers ----------

function inPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    // Indexing, not destructuring: this runs for every pixel of the colour sample, and is ~10x faster.
    const xi = poly[i][0];
    const yi = poly[i][1];
    const xj = poly[j][0];
    const yj = poly[j][1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function distToPolygonEdge(x, y, poly) {
  let best = Infinity;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [x1, y1] = poly[j];
    const [x2, y2] = poly[i];
    const dx = x2 - x1;
    const dy = y2 - y1;
    const t = Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy || 1)));
    best = Math.min(best, Math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)));
  }
  return best;
}

const smoothstep = (a, b, x) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

// ---------- colour: sRGB <-> CIE LAB (same maths as the shader) ----------

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function rgbToLab(r, g, b) {
  const R = srgbToLinear(r);
  const G = srgbToLinear(g);
  const B = srgbToLinear(b);
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const fx = f((0.4124 * R + 0.3576 * G + 0.1805 * B) / 0.95047);
  const fy = f(0.2126 * R + 0.7152 * G + 0.0722 * B);
  const fz = f((0.0193 * R + 0.1192 * G + 0.9505 * B) / 1.08883);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

// Mean and spread of LAB colour inside a polygon of an RGBA image.
function labStats(data, width, height, poly) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of poly) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const sum = [0, 0, 0];
  const sq = [0, 0, 0];
  let n = 0;
  for (let y = Math.max(0, Math.floor(minY)); y < Math.min(height, Math.ceil(maxY)); y += 1) {
    for (let x = Math.max(0, Math.floor(minX)); x < Math.min(width, Math.ceil(maxX)); x += 1) {
      if (!inPolygon(x + 0.5, y + 0.5, poly)) continue;
      const i = (y * width + x) * 4;
      const lab = rgbToLab(data[i] / 255, data[i + 1] / 255, data[i + 2] / 255);
      for (let c = 0; c < 3; c += 1) {
        sum[c] += lab[c];
        sq[c] += lab[c] * lab[c];
      }
      n += 1;
    }
  }
  if (n < 30) return null;
  const mean = sum.map((s) => s / n);
  const std = sq.map((s, c) => Math.max(1, Math.sqrt(Math.max(0, s / n - mean[c] * mean[c]))));
  return { mean, std };
}

// ---------- glasses ----------
// Glasses show as thin dark lines: the "black top-hat" (grey closing minus the image) picks out
// dark structures thinner than the filter, like rims, but not broad shading.

// Max (dilate) or min (erode) over a (2r+1)² square, separable, on a w×h Float32Array.
function rankFilter(src, w, h, r, dilate) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let v = src[y * w + x];
      for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k += 1) {
        const s = src[y * w + k];
        if (dilate ? s > v : s < v) v = s;
      }
      tmp[y * w + x] = v;
    }
  }
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let v = tmp[y * w + x];
      for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k += 1) {
        const s = tmp[k * w + x];
        if (dilate ? s > v : s < v) v = s;
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

// Glasses score (see glassesScore) where glasses start to count, and where they fully count.
const GLASSES_LO = 10;
const GLASSES_HI = 14;

const closing = (src, w, h, r) => rankFilter(rankFilter(src, w, h, r, true), w, h, r, false);

function centroid(ids, pt) {
  let x = 0;
  let y = 0;
  for (const i of ids) {
    const p = pt(i);
    x += p[0];
    y += p[1];
  }
  return [x / ids.length, y / ids.length];
}

// The eyes' frame: centres, eye distance, and unit axes along and across the eye line.
function eyeFrame(pt) {
  const r = centroid(RIGHT_EYE, pt);
  const l = centroid(LEFT_EYE, pt);
  const ed = Math.hypot(l[0] - r[0], l[1] - r[1]) || 1;
  const ax = [(l[0] - r[0]) / ed, (l[1] - r[1]) / ed];
  return { r, l, ed, ax, up: [ax[1], -ax[0]], mid: [(r[0] + l[0]) / 2, (r[1] + l[1]) / 2] };
}

// How strongly a dark line crosses the nose bridge (glasses), minus the forehead's own fine texture
// (wrinkles). Reads an RGBA patch `data` (patchW wide) whose top-left is (ox, oy) in image pixels.
function glassesScore(data, patchW, patchH, ox, oy, eyes) {
  const { ed, ax, up, mid } = eyes;
  const r = Math.max(1, Math.round(0.06 * ed));
  const L = new Float32Array(patchW * patchH);
  for (let i = 0; i < patchW * patchH; i += 1) L[i] = (0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2]) / 2.55;
  const close = closing(L, patchW, patchH, r);
  // s: along the eye line, t: up the face (both in eye distances from the eyes' midpoint).
  const hat = (s, t) => {
    const x = Math.round(mid[0] + (s * ax[0] + t * up[0]) * ed - ox);
    const y = Math.round(mid[1] + (s * ax[1] + t * up[1]) * ed - oy);
    if (x < 0 || y < 0 || x >= patchW || y >= patchH) return 0;
    return close[y * patchW + x] - L[y * patchW + x];
  };
  const step = 1 / ed;
  let bridge = 0;
  for (let t = -0.15; t <= 0.35; t += step) {
    let sum = 0;
    let n = 0;
    for (let s = -0.12; s <= 0.12; s += step) {
      sum += hat(s, t);
      n += 1;
    }
    bridge = Math.max(bridge, sum / Math.max(1, n));
  }
  let tex = 0;
  let n = 0;
  for (let t = 0.6; t <= 0.9; t += step * 2) {
    for (let s = -0.3; s <= 0.3; s += step * 2) {
      tex += hat(s, t);
      n += 1;
    }
  }
  return bridge - tex / Math.max(1, n);
}

// Pixel box (clamped) around the eye midpoint: `side` eye distances left/right, `above`/`below` up/down.
function glassesBox(eyes, width, height, side, above, below) {
  const { ed, mid } = eyes;
  const x0 = Math.max(0, Math.floor(mid[0] - side * ed));
  const y0 = Math.max(0, Math.floor(mid[1] - above * ed));
  const x1 = Math.min(width, Math.ceil(mid[0] + side * ed));
  const y1 = Math.min(height, Math.ceil(mid[1] + below * ed));
  return { x0, y0, w: Math.max(1, x1 - x0), h: Math.max(1, y1 - y0) };
}

// Glasses score of the face in a canvas (see glassesScore). Big faces are checked at reduced size
// (eye distance about 100 px) so it stays a few milliseconds.
let glassesCtx = null;
function detectGlasses(canvas, width, height, eyes) {
  const { x0, y0, w, h } = glassesBox(eyes, width, height, 0.45, 1.05, 0.3);
  const k = Math.min(1, 100 / eyes.ed);
  const pw = Math.max(1, Math.round(w * k));
  const ph = Math.max(1, Math.round(h * k));
  if (!glassesCtx) glassesCtx = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  if (glassesCtx.canvas.width < pw) glassesCtx.canvas.width = pw;
  if (glassesCtx.canvas.height < ph) glassesCtx.canvas.height = ph;
  glassesCtx.drawImage(canvas, x0, y0, w, h, 0, 0, pw, ph);
  const patch = glassesCtx.getImageData(0, 0, pw, ph).data;
  const small = { ...eyes, ed: eyes.ed * k, mid: [(eyes.mid[0] - x0) * k, (eyes.mid[1] - y0) * k] };
  return glassesScore(patch, pw, ph, 0, 0, small);
}

// ---------- smoothing: steady when still, no added delay when the head moves ----------
// MediaPipe already smooths its points a little in video mode, and filtering each point on its own
// adds more delay whenever the head moves. So each frame's points are split into the head's move
// (position, roll and size: a similarity fit to a template, made steady by all 468 points together)
// and the face's shape in the head's own frame. Only the shape is smoothed, with a one-euro filter per
// point (steady when still, quick when the shape changes, as when you talk or turn); the head's move
// passes straight through. Shape speeds are in eye distances per second, so near and far faces match.

function eyeDistance(p) {
  const rx = (p[33 * 2] + p[133 * 2]) / 2;
  const ry = (p[33 * 2 + 1] + p[133 * 2 + 1]) / 2;
  const lx = (p[362 * 2] + p[263 * 2]) / 2;
  const ly = (p[362 * 2 + 1] + p[263 * 2 + 1]) / 2;
  return Math.hypot(lx - rx, ly - ry);
}

class ShapeFilter {
  constructor() {
    this.tpl = null;
  }

  reset() {
    this.tpl = null;
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(values, tMs) {
    const len = values.length;
    const n = len / 2;
    let cx = 0;
    let cy = 0;
    for (let i = 0; i < len; i += 2) {
      cx += values[i];
      cy += values[i + 1];
    }
    cx /= n;
    cy /= n;
    const first = !this.tpl || this.tpl.length !== len;
    if (first) {
      // The template: this first frame, centred, with an eye distance of 1.
      const eye = eyeDistance(values) || 1;
      this.tpl = new Float64Array(len);
      this.tplNorm = 0;
      for (let i = 0; i < len; i += 2) {
        this.tpl[i] = (values[i] - cx) / eye;
        this.tpl[i + 1] = (values[i + 1] - cy) / eye;
        this.tplNorm += this.tpl[i] * this.tpl[i] + this.tpl[i + 1] * this.tpl[i + 1];
      }
      this.shape = new Float64Array(len);
      this.x = new Float64Array(len);
      this.dx = new Float64Array(len);
      this.out = new Float32Array(len);
    }
    // The head's move: points - centre = [a -b; b a] * template (least squares).
    let sa = 0;
    let sb = 0;
    for (let i = 0; i < len; i += 2) {
      const qx = this.tpl[i];
      const qy = this.tpl[i + 1];
      const px = values[i] - cx;
      const py = values[i + 1] - cy;
      sa += qx * px + qy * py;
      sb += qx * py - qy * px;
    }
    const a = sa / this.tplNorm;
    const b = sb / this.tplNorm;
    const s2 = a * a + b * b || 1;
    // The face's shape in the head's frame.
    const shape = this.shape;
    for (let i = 0; i < len; i += 2) {
      const px = values[i] - cx;
      const py = values[i + 1] - cy;
      shape[i] = (a * px + b * py) / s2;
      shape[i + 1] = (-b * px + a * py) / s2;
    }
    if (first) {
      this.x.set(shape);
    } else {
      const dt = Math.max(1e-3, (tMs - this.t) / 1000);
      const ad = ShapeFilter.alpha(TUNE.dCutoff, dt);
      for (let i = 0; i < len; i += 2) {
        this.dx[i] = ad * ((shape[i] - this.x[i]) / dt) + (1 - ad) * this.dx[i];
        this.dx[i + 1] = ad * ((shape[i + 1] - this.x[i + 1]) / dt) + (1 - ad) * this.dx[i + 1];
        const speed = Math.sqrt(this.dx[i] * this.dx[i] + this.dx[i + 1] * this.dx[i + 1]);
        const al = ShapeFilter.alpha(TUNE.minCutoff + TUNE.beta * speed, dt);
        this.x[i] = al * shape[i] + (1 - al) * this.x[i];
        this.x[i + 1] = al * shape[i + 1] + (1 - al) * this.x[i + 1];
      }
    }
    this.t = tMs;
    // Smoothed shape, placed by this frame's head move.
    for (let i = 0; i < len; i += 2) {
      this.out[i] = cx + a * this.x[i] - b * this.x[i + 1];
      this.out[i + 1] = cy + b * this.x[i] + a * this.x[i + 1];
    }
    return this.out;
  }
}

// The points to draw each frame: smoothed while tracked; when tracking drops out, the last good
// points are kept for a moment while the face fades out, and a face that (re)appears fades in.
export class PointSmoother {
  constructor() {
    this.shapeFilter = new ShapeFilter();
    this.reset();
  }

  reset() {
    this.shapeFilter.reset();
    this.points = null;
    this.strength = 0; // face opacity, 0..1
    this.seenAt = -Infinity;
    this.lastT = null;
  }

  // raw: this frame's tracked points (px) or null when no face was found; t: frame time (ms).
  update(raw, t) {
    const dt = this.lastT === null ? 1000 / 30 : Math.min(100, Math.max(0, t - this.lastT));
    this.lastT = t;
    if (raw) {
      this.points = this.shapeFilter.filter(raw, t);
      this.seenAt = t;
      this.strength = TUNE.fadeInMs > 0 ? Math.min(1, this.strength + dt / TUNE.fadeInMs) : 1;
    } else if (this.points && t - this.seenAt < TUNE.holdMs) {
      this.strength = TUNE.fadeOutMs > 0 ? Math.max(0, this.strength - dt / TUNE.fadeOutMs) : 0;
    } else {
      this.shapeFilter.reset();
      this.points = null;
      this.strength = 0;
    }
    return this.points;
  }
}

// ---------- WebGL ----------

const VERTEX_QUAD = `#version 300 es
in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = vec2(aPos.x * 0.5 + 0.5, 0.5 - aPos.y * 0.5);
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

const FRAGMENT_QUAD = `#version 300 es
precision highp float;
uniform sampler2D uFrame;
in vec2 vUV;
out vec4 outColor;
void main() { outColor = vec4(texture(uFrame, vUV).rgb, 1.0); }`;

const VERTEX_FACE = `#version 300 es
in vec2 aPos;     // target position in pixels
in vec2 aSrcUV;   // where this point is in the photo (0..1)
in float aAlpha;  // edge feather x turned-away fade
uniform vec2 uViewport;
out vec2 vSrcUV;
out float vAlpha;
void main() {
  vSrcUV = aSrcUV;
  vAlpha = aAlpha;
  gl_Position = vec4(aPos.x / uViewport.x * 2.0 - 1.0, 1.0 - aPos.y / uViewport.y * 2.0, 0.0, 1.0);
}`;

const FRAGMENT_FACE = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform sampler2D uFrame;
uniform vec2 uViewport;
uniform vec3 uSrcMean, uSrcStd, uDstMean, uDstStd;
uniform float uSrcLod, uFrameLod, uShading, uDetail, uStrength;
uniform float uCamGlasses, uGlassEd, uGlassLod;
uniform vec2 uGlassR, uGlassL, uGlassAx; // your eye centres and eye line, in output pixels
in vec2 vSrcUV;
in float vAlpha;
out vec4 outColor;

vec3 toLinear(vec3 c) { return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), step(0.04045, c)); }
vec3 toSrgb(vec3 c) { c = max(c, 0.0); return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
vec3 f3(vec3 t) { return mix(7.787 * t + 16.0 / 116.0, pow(t, vec3(1.0 / 3.0)), step(0.008856, t)); }
vec3 fi3(vec3 t) { return mix((t - 16.0 / 116.0) / 7.787, t * t * t, step(0.206893, t)); }

vec3 rgb2lab(vec3 c) {
  vec3 l = toLinear(c);
  vec3 xyz = vec3(dot(vec3(0.4124, 0.3576, 0.1805), l) / 0.95047,
                  dot(vec3(0.2126, 0.7152, 0.0722), l),
                  dot(vec3(0.0193, 0.1192, 0.9505), l) / 1.08883);
  vec3 f = f3(xyz);
  return vec3(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z));
}

vec3 lab2rgb(vec3 lab) {
  float fy = (lab.x + 16.0) / 116.0;
  vec3 f = vec3(fy + lab.y / 500.0, fy, fy - lab.z / 200.0);
  vec3 xyz = fi3(f) * vec3(0.95047, 1.0, 1.08883);
  vec3 l = vec3(dot(vec3(3.2406, -1.5372, -0.4986), xyz),
                dot(vec3(-0.9689, 1.8758, 0.0415), xyz),
                dot(vec3(0.0557, -0.2040, 1.0570), xyz));
  return toSrgb(l);
}

// Photo colours re-centred onto the camera's skin tone and lighting.
vec3 transfer(vec3 c) {
  vec3 lab = (rgb2lab(c) - uSrcMean) / uSrcStd * uDstStd + uDstMean;
  return lab2rgb(lab);
}

// Where glasses sit around one eye (less far up than down, so brows stay out), 0..1.
float glassZone(vec2 p, vec2 c) {
  vec2 d = p - c;
  float v = dot(d, vec2(uGlassAx.y, -uGlassAx.x));
  vec2 q = vec2(dot(d, uGlassAx) / 0.6, v / (v > 0.0 ? 0.4 : 0.48)) / uGlassEd;
  return 1.0 - smoothstep(0.8, 1.0, length(q));
}

// Both eyes, but only low over the nose bridge (the frown lines above it stay out).
float glassesArea(vec2 p) {
  vec2 d = (p - 0.5 * (uGlassR + uGlassL)) / uGlassEd;
  float across = abs(dot(d, uGlassAx));
  float up = dot(d, vec2(uGlassAx.y, -uGlassAx.x));
  float bridge = (1.0 - smoothstep(0.15, 0.3, across)) * smoothstep(0.1, 0.18, up);
  return max(glassZone(p, uGlassR), glassZone(p, uGlassL)) * (1.0 - bridge);
}

void main() {
  vec2 fuv = vec2(gl_FragCoord.x / uViewport.x, 1.0 - gl_FragCoord.y / uViewport.y);
  vec3 src = transfer(texture(uSrc, vSrcUV).rgb);
  vec3 srcLow = transfer(textureLod(uSrc, vSrcUV, uSrcLod).rgb);
  vec3 frame = texture(uFrame, fuv).rgb;
  vec3 frameLow = textureLod(uFrame, fuv, uFrameLod).rgb;
  // Photo detail + the room's shading on your face + your live skin highlights.
  vec3 col = src + (frameLow - srcLow) * uShading + (frame - frameLow) * uDetail;
  float alpha = vAlpha;
  if (uCamGlasses > 0.0) {
    // Your glasses rims (thin dark lines in the glasses area) stay on top.
    vec2 fpx = vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y);
    float z = glassesArea(fpx);
    if (z > 0.0) {
      float dark = dot(textureLod(uFrame, fuv, uGlassLod).rgb - frame, vec3(0.299, 0.587, 0.114));
      alpha *= 1.0 - uCamGlasses * z * smoothstep(0.07, 0.15, dark);
    }
  }
  outColor = vec4(clamp(col, 0.0, 1.0), alpha * uStrength);
}`;

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader error: ${log}`);
  }
  return shader;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`Shader link error: ${gl.getProgramInfoLog(p)}`);
  return p;
}

function makeTexture(gl) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

// ---------- the engine ----------

export class LiteFaceAI {
  constructor({ api, onState, onError }) {
    this.api = api;
    this.onState = onState;
    this.onError = onError;
    this.state = 'off';
    this.target = null;
    this.dirty = false;
    this.running = false;
    this.ready = false;
    this.delegate = '';
    this.appliedRef = null;
    this.source = null; // { width, height, points: Float32Array (px), tris: Uint16Array, feather, stats }
    this.dstStats = null;
    this.camGlasses = 0;
    this.frame = 0;
    this.lastTs = 0;
    this.lastResultAt = 0;
    this.faceFound = false;
    this.stats = { ms: 0, fps: 0 };
    this.times = [];
    this.smoother = new PointSmoother();
    this.raw = new Float32Array(MESH_POINTS * 2); // this frame's tracked points before smoothing (px)
    this.points = null; // the points drawn this frame

    this.input = document.createElement('canvas');
    this.input.width = output.width;
    this.input.height = output.height;
    this.inputCtx = this.input.getContext('2d', { alpha: false, willReadFrequently: true });
    this.small = document.createElement('canvas');
    this.small.width = 320;
    this.small.height = 180;
    this.smallCtx = this.small.getContext('2d', { willReadFrequently: true });

    this.canvas = document.createElement('canvas');
    this.canvas.width = output.width;
    this.canvas.height = output.height;
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState(state);
  }

  get live() {
    return this.state === 'generating' && performance.now() - this.lastResultAt < FRESH_MS;
  }

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
      if (!this.ready) {
        this.setState('starting');
        await this.init();
      }
      if (this.target !== target) return;
      if (target.refKey !== this.appliedRef) {
        this.source = null;
        this.appliedRef = target.refKey;
        if (target.refBlob) {
          this.setState('loading');
          this.source = await this.prepareSource(target.refBlob);
          if (!this.source) this.appliedRef = `missing:${target.refKey}`;
        }
      }
      if (!target.refBlob) this.setState('needs-reference');
      else if (!this.source) this.setState('no-face');
      else this.setState('generating');
    } catch (err) {
      this.setState('error');
      this.onError(`Lite AI: ${(err && err.message) || err}`);
    }
  }

  async init() {
    // The face is drawn with WebGL 2; every Windows 10/11 PC with a working graphics driver has it.
    const probe = document.createElement('canvas').getContext('webgl2');
    if (!probe) throw new Error('This computer has no WebGL 2 graphics support. Update the graphics driver.');
    const wasm = await this.api.readAsset('mediapipe/vision_wasm_internal.wasm');
    const model = await this.api.readAsset('mediapipe/face_landmarker.task');
    const fileset = {
      wasmLoaderPath: new URL('mediapipe/vision_wasm_internal.js', document.baseURI).href,
      wasmBinaryPath: URL.createObjectURL(new Blob([wasm], { type: 'application/wasm' })),
    };
    const make = async (runningMode, delegate) =>
      FaceLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: new Uint8Array(model), delegate },
        runningMode,
        numFaces: 1,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    // The GPU path is fastest; the CPU path works everywhere.
    try {
      this.videoLm = await make('VIDEO', 'GPU');
      this.delegate = 'GPU';
    } catch {
      this.videoLm = await make('VIDEO', 'CPU');
      this.delegate = 'CPU';
    }
    this.imageLm = await make('IMAGE', 'CPU');
    this.initGl();
    this.ready = true;
  }

  initGl() {
    const gl = this.canvas.getContext('webgl2', { alpha: false, preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error('This computer has no WebGL 2 support');
    this.gl = gl;
    this.quadProgram = program(gl, VERTEX_QUAD, FRAGMENT_QUAD);
    this.faceProgram = program(gl, VERTEX_FACE, FRAGMENT_FACE);
    this.quadBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    this.faceBuffer = gl.createBuffer();
    this.frameTex = makeTexture(gl);
    this.srcTex = makeTexture(gl);
    this.loc = {
      qPos: gl.getAttribLocation(this.quadProgram, 'aPos'),
      qFrame: gl.getUniformLocation(this.quadProgram, 'uFrame'),
      fPos: gl.getAttribLocation(this.faceProgram, 'aPos'),
      fSrcUV: gl.getAttribLocation(this.faceProgram, 'aSrcUV'),
      fAlpha: gl.getAttribLocation(this.faceProgram, 'aAlpha'),
    };
    for (const name of ['uSrc', 'uFrame', 'uViewport', 'uSrcMean', 'uSrcStd', 'uDstMean', 'uDstStd', 'uSrcLod', 'uFrameLod', 'uShading', 'uDetail', 'uStrength', 'uCamGlasses', 'uGlassEd', 'uGlassLod', 'uGlassR', 'uGlassL', 'uGlassAx']) {
      this.loc[name] = gl.getUniformLocation(this.faceProgram, name);
    }
  }

  // Finds the face in the photo once and builds the fixed mesh used for every frame.
  async prepareSource(blob) {
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, 1280 / Math.max(bitmap.width, bitmap.height));
    const width = Math.round(bitmap.width * scale);
    const height = Math.round(bitmap.height * scale);
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();

    let result = this.imageLm.detect(canvas);
    let pad = 0;
    if (!result.faceLandmarks.length) {
      // Tight close-ups detect better with a border around them.
      pad = Math.round(Math.max(width, height) * 0.25);
      const padded = document.createElement('canvas');
      padded.width = width + pad * 2;
      padded.height = height + pad * 2;
      const pctx = padded.getContext('2d');
      pctx.fillStyle = '#000';
      pctx.fillRect(0, 0, padded.width, padded.height);
      pctx.drawImage(canvas, pad, pad);
      result = this.imageLm.detect(padded);
      if (!result.faceLandmarks.length) return null;
      const lm = result.faceLandmarks[0];
      result = { faceLandmarks: [lm.map((p) => ({ x: (p.x * padded.width - pad) / width, y: (p.y * padded.height - pad) / height }))] };
    }

    const lm = result.faceLandmarks[0];
    const points = new Float32Array(MESH_POINTS * 2);
    for (let i = 0; i < MESH_POINTS; i += 1) {
      points[i * 2] = lm[i].x * width;
      points[i * 2 + 1] = lm[i].y * height;
    }
    const pt = (i) => [points[i * 2], points[i * 2 + 1]];
    const oval = FACE_OVAL.map(pt);
    const holes = HOLES.map((ring) => ring.map(pt));

    // A triangle joining the upper and lower lid/lip would stretch across your eye or mouth
    // when it opens wider than the photo's (closed or narrow in the photo).
    const side = new Int8Array(MESH_POINTS * HOLES.length);
    HOLE_SIDES.forEach(([upper, lower], k) => {
      for (const v of upper) side[k * MESH_POINTS + v] = 1;
      for (const v of lower) side[k * MESH_POINTS + v] = 2;
    });
    const spansHole = (a, b, c) =>
      HOLES.some((_, k) => {
        const m = side[k * MESH_POINTS + a] | side[k * MESH_POINTS + b] | side[k * MESH_POINTS + c];
        return m === 3;
      });

    // Fixed topology from the photo: never re-triangulated, so the mesh can't flicker.
    const del = new Delaunator(points);
    const tris = [];
    for (let t = 0; t < del.triangles.length; t += 3) {
      const [a, b, c] = [del.triangles[t], del.triangles[t + 1], del.triangles[t + 2]];
      const cx = (points[a * 2] + points[b * 2] + points[c * 2]) / 3;
      const cy = (points[a * 2 + 1] + points[b * 2 + 1] + points[c * 2 + 1]) / 3;
      if (!inPolygon(cx, cy, oval)) continue; // outside the face
      if (holes.some((h) => inPolygon(cx, cy, h))) continue; // your real eyes and mouth show through
      if (TUNE.spanCut && spansHole(a, b, c)) continue;
      tris.push(a, b, c);
    }

    // Soft edges: fade towards the face outline and around the eye and mouth holes.
    const faceWidth = Math.max(...oval.map((p) => p[0])) - Math.min(...oval.map((p) => p[0]));
    const feather = new Float32Array(MESH_POINTS);
    for (let i = 0; i < MESH_POINTS; i += 1) {
      const [x, y] = pt(i);
      const edge = inPolygon(x, y, oval) ? distToPolygonEdge(x, y, oval) / faceWidth : 0;
      let a = smoothstep(0.0, Math.max(1e-4, TUNE.feather), edge);
      holes.forEach((h, k) => {
        const mouth = k === 2;
        const d = distToPolygonEdge(x, y, h) / faceWidth;
        const w = Math.max(1e-4, mouth ? TUNE.mouthFeather : TUNE.holeFeather);
        a = Math.min(a, mouth ? smoothstep(0.0, w, d) : TUNE.eyeEdge + (1 - TUNE.eyeEdge) * smoothstep(0.0, w, d));
      });
      feather[i] = a;
    }
    const srcAreas = new Float32Array(tris.length / 3);
    for (let t = 0; t < tris.length; t += 3) srcAreas[t / 3] = triArea(points, tris[t], tris[t + 1], tris[t + 2]);

    const stats = labStats(ctx.getImageData(0, 0, width, height).data, width, height, oval) || {
      mean: [60, 10, 15],
      std: [15, 5, 7],
    };

    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    gl.generateMipmap(gl.TEXTURE_2D);

    this.smoother.reset();
    this.dstStats = null;
    return { width, height, points, tris: Uint16Array.from(tris), feather, srcAreas, stats, faceWidth };
  }

  // Called by the compositor every frame. frameTime (ms) is when the frame was captured; it defaults
  // to now (the benchmark passes simulated 30 fps camera timestamps).
  tick(drawSource, frameTime) {
    if (this.state !== 'generating' || !this.ready || !this.source) return;
    const started = performance.now();
    const t = frameTime ?? started;
    if (this.input.width !== output.width || this.input.height !== output.height) {
      // Picture shape changed: follow it and restart smoothing (all points just moved).
      this.input.width = output.width;
      this.input.height = output.height;
      this.canvas.width = output.width;
      this.canvas.height = output.height;
      this.smoother.reset();
      this.dstStats = null;
    }
    drawSource(this.inputCtx, output.width, output.height);
    const ts = Math.max(this.lastTs + 1, Math.floor(t));
    this.lastTs = ts;
    let landmarks = null;
    try {
      const result = this.videoLm.detectForVideo(this.input, ts);
      landmarks = result.faceLandmarks && result.faceLandmarks[0];
    } catch (err) {
      this.setState('error');
      this.onError(`Lite AI tracking failed: ${(err && err.message) || err}`);
      return;
    }

    const raw = this.raw;
    if (landmarks) {
      for (let i = 0; i < MESH_POINTS; i += 1) {
        raw[i * 2] = landmarks[i].x * output.width;
        raw[i * 2 + 1] = landmarks[i].y * output.height;
      }
    }
    // While tracking is lost for a moment, the last good points stay (fading out).
    const points = this.smoother.update(landmarks ? raw : null, t);
    if (landmarks && (this.frame % 4 === 0 || !this.dstStats)) this.updateDstStats(points);
    this.points = points;
    this.faceFound = Boolean(landmarks);
    this.render(points, this.smoother.strength);
    this.frame += 1;

    const now = performance.now();
    this.lastResultAt = now;
    this.stats.ms = Math.round(now - started);
    this.times.push(now);
    while (this.times.length && now - this.times[0] > 1000) this.times.shift();
    this.stats.fps = this.times.length;
  }

  updateDstStats(points) {
    // Keep the colour sample canvas the same shape as the picture.
    const smallW = output.width >= output.height ? 320 : 180;
    const smallH = output.width >= output.height ? 180 : 320;
    if (this.small.width !== smallW || this.small.height !== smallH) {
      this.small.width = smallW;
      this.small.height = smallH;
    }
    const sx = this.small.width / output.width;
    const sy = this.small.height / output.height;
    this.smallCtx.drawImage(this.input, 0, 0, this.small.width, this.small.height);
    const data = this.smallCtx.getImageData(0, 0, this.small.width, this.small.height).data;
    const oval = FACE_OVAL.map((i) => [points[i * 2] * sx, points[i * 2 + 1] * sy]);
    const stats = labStats(data, this.small.width, this.small.height, oval);
    if (!TUNE.glassesCam) this.camGlasses = 0;
    else if (!this.dstStats || this.frame % 16 === 0) {
      // Your own glasses (checked from the camera picture about twice a second, smoothed so it
      // doesn't flicker).
      this.camGlassesScore = detectGlasses(this.input, output.width, output.height, eyeFrame((i) => [points[i * 2], points[i * 2 + 1]]));
      const g = smoothstep(GLASSES_LO, GLASSES_HI, this.camGlassesScore);
      this.camGlasses = this.dstStats ? this.camGlasses * 0.5 + g * 0.5 : g;
    }
    if (stats) this.dstStats = stats;
  }

  render(points, strength = 1) {
    const gl = this.gl;
    gl.viewport(0, 0, output.width, output.height);
    gl.disable(gl.BLEND);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.frameTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.input);
    gl.generateMipmap(gl.TEXTURE_2D);

    // Your camera picture first.
    gl.useProgram(this.quadProgram);
    gl.uniform1i(this.loc.qFrame, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadBuffer);
    gl.enableVertexAttribArray(this.loc.qPos);
    gl.vertexAttribPointer(this.loc.qPos, 2, gl.FLOAT, false, 0, 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.disableVertexAttribArray(this.loc.qPos);
    if (!points || !this.dstStats || strength <= 0) return;

    // Then the photo's face, warped onto yours.
    const src = this.source;
    const tris = src.tris;
    const triCount = tris.length / 3;
    let minX = Infinity;
    let maxX = -Infinity;
    for (let k = 0; k < FACE_OVAL.length; k += 1) {
      const x = points[FACE_OVAL[k] * 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
    }
    const targetWidth = maxX - minX;
    const scale2 = (targetWidth / src.faceWidth) ** 2;
    // One vertex array per photo, reused every frame: x, y, photo u, v (fixed), alpha.
    if (!src.verts) {
      src.verts = new Float32Array(triCount * 3 * 5);
      for (let v = 0; v < triCount * 3; v += 1) {
        src.verts[v * 5 + 2] = src.points[tris[v] * 2] / src.width;
        src.verts[v * 5 + 3] = src.points[tris[v] * 2 + 1] / src.height;
      }
    }
    const verts = src.verts;
    for (let t = 0; t < triCount; t += 1) {
      const a = tris[t * 3];
      const b = tris[t * 3 + 1];
      const c = tris[t * 3 + 2];
      // Triangles that shrink or flip are turning away from the camera: fade them out.
      const ratio = triArea(points, a, b, c) / (src.srcAreas[t] * scale2 || 1);
      const facing = smoothstep(TUNE.turnLo, TUNE.turnHi, ratio);
      const o = t * 15;
      verts[o] = points[a * 2];
      verts[o + 1] = points[a * 2 + 1];
      verts[o + 4] = src.feather[a] * facing;
      verts[o + 5] = points[b * 2];
      verts[o + 6] = points[b * 2 + 1];
      verts[o + 9] = src.feather[b] * facing;
      verts[o + 10] = points[c * 2];
      verts[o + 11] = points[c * 2 + 1];
      verts[o + 14] = src.feather[c] * facing;
    }

    gl.useProgram(this.faceProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(this.loc.uSrc, 0);
    gl.uniform1i(this.loc.uFrame, 1);
    gl.uniform2f(this.loc.uViewport, output.width, output.height);
    gl.uniform3fv(this.loc.uSrcMean, src.stats.mean);
    gl.uniform3fv(this.loc.uSrcStd, src.stats.std);
    gl.uniform3fv(this.loc.uDstMean, this.dstStats.mean);
    gl.uniform3fv(this.loc.uDstStd, this.dstStats.std.map((d, c) => d + (src.stats.std[c] - d) * TUNE.contrast));
    // Blur radius about 1/BLUR_DIV of the face, in each image's own scale: only broad lighting moves over.
    gl.uniform1f(this.loc.uSrcLod, Math.max(0, Math.log2(src.faceWidth / TUNE.blurDiv)));
    gl.uniform1f(this.loc.uFrameLod, Math.max(0, Math.log2(targetWidth / TUNE.blurDiv)));
    gl.uniform1f(this.loc.uShading, TUNE.shading);
    gl.uniform1f(this.loc.uDetail, TUNE.detail);
    gl.uniform1f(this.loc.uStrength, strength);
    const camGlasses = this.camGlasses > 0.05 ? this.camGlasses : 0;
    gl.uniform1f(this.loc.uCamGlasses, camGlasses);
    if (camGlasses) {
      const eyes = eyeFrame((i) => [points[i * 2], points[i * 2 + 1]]);
      gl.uniform2fv(this.loc.uGlassR, eyes.r);
      gl.uniform2fv(this.loc.uGlassL, eyes.l);
      gl.uniform2fv(this.loc.uGlassAx, eyes.ax);
      gl.uniform1f(this.loc.uGlassEd, eyes.ed);
      gl.uniform1f(this.loc.uGlassLod, Math.max(0, Math.log2(0.15 * eyes.ed)));
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.faceBuffer);
    if (this.faceBufferLength !== verts.length) {
      gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
      this.faceBufferLength = verts.length;
    } else {
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, verts);
    }
    const stride = 5 * 4;
    gl.enableVertexAttribArray(this.loc.fPos);
    gl.vertexAttribPointer(this.loc.fPos, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.loc.fSrcUV);
    gl.vertexAttribPointer(this.loc.fSrcUV, 2, gl.FLOAT, false, stride, 8);
    gl.enableVertexAttribArray(this.loc.fAlpha);
    gl.vertexAttribPointer(this.loc.fAlpha, 1, gl.FLOAT, false, stride, 16);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, triCount * 3);
    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(this.loc.fPos);
    gl.disableVertexAttribArray(this.loc.fSrcUV);
    gl.disableVertexAttribArray(this.loc.fAlpha);
  }
}

// Signed area: negative when the triangle has flipped over.
function triArea(p, a, b, c) {
  return ((p[b * 2] - p[a * 2]) * (p[c * 2 + 1] - p[a * 2 + 1]) - (p[c * 2] - p[a * 2]) * (p[b * 2 + 1] - p[a * 2 + 1])) / 2;
}
