// Lite engine: "wear" the face from a photo with no generative AI, so it runs on almost any PC.
//
//   1. Track 478 face points with MediaPipe (small, runs on the CPU). The face's shape is smoothed in
//      the head's own frame, so the head's moves aren't delayed; brief tracking losses fade, not pop.
//   2. Warp the photo's face onto yours with a fixed triangle mesh built once from the photo,
//      leaving holes for your real eyes and mouth so blinks, speech and teeth stay yours.
//      Your face points are first moved to the photo's face shape (aligned to your head pose, minus
//      your expressions), and your camera picture is warped the same way, eyes and mouth included.
//   3. In a WebGL shader: match the photo's colours to your camera in LAB space, take the room's
//      shading from your camera (frequency separation), add back your skin highlights, and feather
//      the edges. Parts turned away (in your camera or in the photo) fade out instead of smearing. If you
//      wear glasses, their rims stay on top of the photo's face.
import { FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';
import Delaunator from 'delaunator';
import { output } from './state.js';

const FRESH_MS = 1000;
// Tunable look of the Lite engine (the benchmark adjusts these; the app uses the defaults).
//   shading:  how much of the camera's broad lighting is laid over the photo's face
//   detail:   how much of the camera's fine skin detail/highlights is added back
//   blurDiv:  lighting blur radius = face width / blurDiv
//   contrast: 0 = take the camera's colour contrast, 1 = keep the photo's
//   feather:  soft edge at the face outline (cheeks and jaw), as a share of face width
//   featherTop: the same up the forehead, towards the hairline
//   featherLive: 1 = the outline's soft edge is worked out per pixel where the face is drawn
//             (0 = from the photo's own outline at the mesh points, jagged on a turned cheek)
//   frontalMesh: 1 = the photo's triangle mesh is built on its face turned straight on, so a cheek
//             the photo shows turned away still gets triangles (no gaps when your cheek faces you)
//   beyondFeather: with frontalMesh, mesh points the photo shows beyond its own outline (hair or
//             background may be there) fade out over this share of the face width
//   holeFeather: soft edge around the eye holes, as a share of face width
//   mouthFeather: soft edge around the mouth hole, as a share of face width
//   eyeEdge:  photo opacity right at the eye opening (0 = fade to your eye, 1 = hard edge)
//   glassesCam: 1 = keep the rims of your own glasses on top of the photo's face (when detected)
//   spanCut:  1 = also cut triangles that join the upper and lower lid/lip (closed eyes/mouth in the photo)
//   minCutoff, beta, dCutoff: one-euro smoothing of the face's shape in the head's own frame (Hz when
//             still; extra Hz per eye distance/s of shape change; Hz for the speed estimate)
//   turnLo, turnHi: the face fades out where its size falls from turnHi to turnLo of the photo's
//                   (it is turning away from the camera); measured per point, averaged over its
//                   triangles and smoothed over the mesh for turnSmooth rounds (0 = per triangle, the
//                   old way, which leaves a saw-tooth of faded triangles along a turned cheek)
//   flipLo, flipHi: with turnSmooth, single triangles still fade as their size falls from flipHi to
//                   flipLo of the photo's (folded or flipped over)
//   stretchLo, stretchHi: the face fades out where it grows from stretchLo to stretchHi times the
//                   photo's (the photo shows that part turned away, so it has too few pixels for it)
//   edgeOnLo, edgeOnHi: the face fades out where it is seen edge-on in the camera or in the photo,
//                   as the view-axis share of its 3D normal (smoothed edgeOnSmooth rounds) falls
//                   from edgeOnHi to edgeOnLo
//   holdMs:   when tracking drops out, keep the last points this long while the face fades out
//   fadeInMs, fadeOutMs: how long the face takes to fade in when found and out when lost
//   depthSmooth: how quickly the tracked depth (used only to align the photo's shape in 3D) follows (0..1 per frame)
//   shape:     0 = keep your face shape, 1 = the photo's face shape (outline, eye spacing, nose, mouth)
//   shape3d:   1 = align the photo to your head in 3D (head turns stay turns), 0 = flat 2D alignment (tears when turned)
//   shapeLips: how much the lips' outer edge takes the photo's shape (lips also move with the face as one piece)
//   shapeMouth: the same for the mouth opening (inner lip edge); low keeps your own opening at rest
//   shapeZone: skin within this share of the face width from the lips is eased from shapeLips to full shape
//   shapeJaw:  1 = the photo's mouth opening is taken out of its shape (an open-mouthed photo doesn't stretch your chin)
//   shapeEyes: 0 = your eyes keep their shape and only move/scale with the photo's eye corners, 1 = full reshape
//   shapeRing: the warp fades out this far outside the face outline (share of the face radius)
//   shapeRest: seconds of averaging for your resting face shape (expressions are changes from it);
//              0 = no averaging (every frame is taken as rest, which also flattens expressions)
//   mask:     1 = leave out what in the photo's face outline isn't skin (hair, glasses, background), so
//             your own (reshaped) camera picture shows there; 0 = draw the whole outline
//   maskLo, maskHi: the photo's skin confidence (0..1) where its face starts to show and fully shows
//   maskStats: 1 = the photo's colour statistics come from its skin only (not its hair or glasses)
//   maskPaintGlasses: 1 = the photo's glasses frames are painted out of the photo (filled from the skin
//                around them); 0 = its glasses are kept, as part of how that person looks (default: the
//                fill leaves smudges around the eyes and glasses are part of the photo's identity)
//   maskGlasses: this much of the photo's glasses' whole area (frames and lenses) is cut, so your
//                camera's own eye area shows there (1 = all of it)
//   maskGlassesSoft: how soft that cut's edge is, as a share of the face size
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
  turnLo: 0.1,
  turnHi: 0.45,
  turnSmooth: 3,
  flipLo: 0,
  flipHi: 0.05,
  stretchLo: 1.5,
  stretchHi: 2.5,
  edgeOnLo: 0.05,
  edgeOnHi: 0.25,
  edgeOnSmooth: 3,
  featherLive: 1,
  featherTop: 0.06,
  frontalMesh: 1,
  beyondFeather: 0.02,
  holdMs: 250,
  fadeInMs: 150,
  fadeOutMs: 200,
  shape: 1,
  shape3d: 1,
  shapeLips: 0.5,
  shapeMouth: 0,
  shapeZone: 0.1,
  shapeJaw: 1,
  shapeEyes: 0,
  shapeRing: 0.35,
  shapeRest: 5,
  depthSmooth: 0.5,
  mask: 1,
  maskLo: 0.3,
  maskHi: 0.8,
  maskStats: 1,
  maskPaintGlasses: 0,
  maskGlasses: 0,
  maskGlassesSoft: 0.04,
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
// Face shape: points that don't move with expression (forehead, temples, nose, eye corners) fix the head pose.
const SHAPE_FIT = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 98, 327, 33, 133, 362, 263, 127, 356, 234, 454, 162, 389, 21, 251, 54, 284, 103, 332, 67, 297, 109, 338];
const OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185];
const EYE_CORNERS = [[33, 133], [263, 362]]; // outer, inner corner of RIGHT_EYE, LEFT_EYE
// The lower lip (all four lip rings) and the mouth corners: they go with the jaw when the mouth opens.
const LOWER_LIP = [146, 91, 181, 84, 17, 314, 405, 321, 375, 77, 90, 180, 85, 16, 315, 404, 320, 307, 96, 89, 179, 86, 15, 316, 403, 319, 325, 95, 88, 178, 87, 14, 317, 402, 318, 324];
const MOUTH_CORNERS = [61, 291, 76, 306, 62, 292, 78, 308];
// Folds in the camera warp: first smooth the moves of a folded triangle's corners (FOLD_SMOOTH_ROUNDS),
// then shrink them (each round they keep FOLD_EASE of their move) until FOLD_ROUNDS.
const FOLD_SMOOTH_ROUNDS = 8;
const FOLD_EASE = 0.75;
const FOLD_ROUNDS = 20;

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

// Mean and spread of LAB colour inside a polygon of an RGBA image (optionally weighted per pixel, 0..1).
function labStats(data, width, height, poly, weight = null) {
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
      const w = weight ? weight(x + 0.5, y + 0.5) : 1;
      if (w <= 0) continue;
      const i = (y * width + x) * 4;
      const lab = rgbToLab(data[i] / 255, data[i + 1] / 255, data[i + 2] / 255);
      for (let c = 0; c < 3; c += 1) {
        sum[c] += lab[c] * w;
        sq[c] += lab[c] * lab[c] * w;
      }
      n += w;
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

// ---------- the photo's skin: its hair, glasses and background left out ----------
// MediaPipe's multiclass selfie segmenter gives each pixel a confidence for background, hair, body skin,
// face skin, clothes and "other" (glasses, jewellery). Brows, eyes, lips, beards and moustaches come out
// as face skin; glasses frames as "other" (the lenses, with the eyes behind them, as face skin).
const SEG_SIZE = 256; // the model's own input size
const SEG_CROP = 2; // the photo is segmented in a square this many face sizes wide, centred on the face
const SKIN_MIN_COVERAGE = 0.3; // below this share of skin inside the face outline, the mask isn't used
const GLASSES_FILL_LO = 0.1; // glasses confidence where the photo starts being painted out, and where fully
const GLASSES_FILL_HI = 0.35;
const GLASSES_CLOSE = 0.12; // gaps up to about this share of the face size (the lenses) are closed over

// Box blur of radius r (separable) on a w×h Float32Array.
function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let s = 0;
      let n = 0;
      for (let k = Math.max(0, x - r); k <= Math.min(w - 1, x + r); k += 1) {
        s += src[y * w + k];
        n += 1;
      }
      tmp[y * w + x] = s / n;
    }
  }
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let s = 0;
      let n = 0;
      for (let k = Math.max(0, y - r); k <= Math.min(h - 1, y + r); k += 1) {
        s += tmp[k * w + x];
        n += 1;
      }
      out[y * w + x] = s / n;
    }
  }
  return out;
}

// Skin confidence (0..255, face + body skin) over a square around the photo's face, and where that
// square is in the photo (px). Null if the segmenter fails.
function segmentSkin(segmenter, canvas, oval) {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const [x, y] of oval) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const side = Math.max(maxX - minX, maxY - minY) * SEG_CROP;
  const x0 = (minX + maxX) / 2 - side / 2;
  const y0 = (minY + maxY) / 2 - side / 2;
  const crop = document.createElement('canvas');
  crop.width = SEG_SIZE;
  crop.height = SEG_SIZE;
  const ctx = crop.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, SEG_SIZE, SEG_SIZE);
  ctx.drawImage(canvas, x0, y0, side, side, 0, 0, SEG_SIZE, SEG_SIZE);
  let res = null;
  try {
    res = segmenter.segment(crop);
    const masks = res.confidenceMasks;
    if (!masks || masks.length < 4) return null;
    const size = masks[0].width;
    const body = masks[2].getAsFloat32Array();
    const face = masks[3].getAsFloat32Array();
    const other = masks.length > 5 ? masks[5].getAsFloat32Array() : new Float32Array(size * size);
    // Glasses frames (dilated by one cell so their dark edges go too) count as skin: they are kept, or
    // painted out of the photo (TUNE.maskPaintGlasses, see paintOut). The glasses' whole area (frames and lenses: the frames closed
    // over, softened) is cut by TUNE.maskGlasses, so your camera's own eye area shows there instead.
    const glasses = rankFilter(other, size, size, 1, true);
    let area = null;
    if (TUNE.maskGlasses > 0) {
      const r = Math.max(1, Math.round((size / SEG_CROP) * GLASSES_CLOSE));
      area = boxBlur(closing(glasses, size, size, r), size, size, Math.max(1, Math.round((size / SEG_CROP) * TUNE.maskGlassesSoft)));
    }
    const data = new Uint8Array(size * size);
    const pure = new Uint8Array(size * size); // skin only (no glasses), for the colour statistics
    for (let i = 0; i < data.length; i += 1) {
      pure[i] = Math.round(Math.min(1, body[i] + face[i]) * 255);
      const s = Math.min(1, body[i] + face[i] + glasses[i]) * (area ? 1 - Math.min(1, area[i] * 1.5) * TUNE.maskGlasses : 1);
      data[i] = Math.round(s * 255);
    }
    return { data, pure, glasses, size, x0, y0, side };
  } catch (err) {
    console.warn(`Lite: photo segmentation failed: ${(err && err.message) || err}`);
    return null;
  } finally {
    if (res) res.close();
  }
}

// The photo's glasses painted out: each glasses pixel is filled from the skin around it (push-pull:
// averages over ever larger blocks, the nearest known colour wins), so its face can be drawn without them.
function paintOut(ctx, width, height, skin) {
  const { size, x0, y0, side, glasses } = skin;
  const bx = Math.max(0, Math.floor(x0));
  const by = Math.max(0, Math.floor(y0));
  const w = Math.min(width, Math.ceil(x0 + side)) - bx;
  const h = Math.min(height, Math.ceil(y0 + side)) - by;
  if (w < 2 || h < 2) return;
  // Level 0: colour premultiplied by how much it is kept (1 = skin, 0 = glasses), soft at the edges.
  const keep0 = new Float32Array(w * h);
  const col0 = new Float32Array(w * h * 3);
  const img = ctx.getImageData(bx, by, w, h);
  const d = img.data;
  const k = size / side;
  let any = false;
  for (let y = 0; y < h; y += 1) {
    const v = Math.min(size - 1, Math.max(0, (by + y + 0.5 - y0) * k - 0.5));
    const v0 = Math.floor(v);
    const v1 = Math.min(size - 1, v0 + 1);
    const fv = v - v0;
    for (let x = 0; x < w; x += 1) {
      const u = Math.min(size - 1, Math.max(0, (bx + x + 0.5 - x0) * k - 0.5));
      const u0 = Math.floor(u);
      const u1 = Math.min(size - 1, u0 + 1);
      const fu = u - u0;
      const g = (glasses[v0 * size + u0] * (1 - fu) + glasses[v0 * size + u1] * fu) * (1 - fv) +
        (glasses[v1 * size + u0] * (1 - fu) + glasses[v1 * size + u1] * fu) * fv;
      const keep = 1 - smoothstep(GLASSES_FILL_LO, GLASSES_FILL_HI, g);
      const i = y * w + x;
      keep0[i] = keep;
      for (let c = 0; c < 3; c += 1) col0[i * 3 + c] = d[i * 4 + c] * keep;
      if (keep < 1) any = true;
    }
  }
  if (!any) return;
  // Pull: halve until one block, each block's weight capped at 1.
  const levels = [{ w, h, keep: keep0, col: col0 }];
  while (levels[levels.length - 1].w > 1 || levels[levels.length - 1].h > 1) {
    const f = levels[levels.length - 1];
    const nw = Math.ceil(f.w / 2);
    const nh = Math.ceil(f.h / 2);
    const keep = new Float32Array(nw * nh);
    const col = new Float32Array(nw * nh * 3);
    for (let y = 0; y < f.h; y += 1) {
      for (let x = 0; x < f.w; x += 1) {
        const i = y * f.w + x;
        const j = (y >> 1) * nw + (x >> 1);
        keep[j] += f.keep[i];
        for (let c = 0; c < 3; c += 1) col[j * 3 + c] += f.col[i * 3 + c];
      }
    }
    for (let j = 0; j < nw * nh; j += 1) {
      const s = keep[j] > 1 ? 1 / keep[j] : 1;
      keep[j] *= s;
      for (let c = 0; c < 3; c += 1) col[j * 3 + c] *= s;
    }
    levels.push({ w: nw, h: nh, keep, col });
  }
  // Push: what a pixel lacks comes from the (already filled) coarser level, interpolated (no blocks).
  for (let l = levels.length - 2; l >= 0; l -= 1) {
    const f = levels[l];
    const p = levels[l + 1];
    for (let j = 0; j < p.w * p.h; j += 1) {
      const pk = p.keep[j] || 1;
      for (let c = 0; c < 3; c += 1) p.col[j * 3 + c] /= pk;
      p.keep[j] = 1;
    }
    for (let y = 0; y < f.h; y += 1) {
      const py = Math.min(p.h - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
      const y0 = Math.floor(py);
      const y1 = Math.min(p.h - 1, y0 + 1);
      const fy = py - y0;
      for (let x = 0; x < f.w; x += 1) {
        const i = y * f.w + x;
        const miss = 1 - f.keep[i];
        f.keep[i] = 1;
        if (miss <= 0) continue;
        const px = Math.min(p.w - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
        const x0 = Math.floor(px);
        const x1 = Math.min(p.w - 1, x0 + 1);
        const fx = px - x0;
        const a = (y0 * p.w + x0) * 3;
        const b = (y0 * p.w + x1) * 3;
        const cc = (y1 * p.w + x0) * 3;
        const dd = (y1 * p.w + x1) * 3;
        for (let c = 0; c < 3; c += 1) {
          const v = (p.col[a + c] * (1 - fx) + p.col[b + c] * fx) * (1 - fy) + (p.col[cc + c] * (1 - fx) + p.col[dd + c] * fx) * fy;
          f.col[i * 3 + c] += miss * v;
        }
      }
    }
  }
  for (let i = 0; i < w * h; i += 1) for (let c = 0; c < 3; c += 1) d[i * 4 + c] = Math.round(col0[i * 3 + c]);
  ctx.putImageData(img, bx, by);
}

// The photo's skin opacity (0..1) at photo pixel (x, y), as the shader computes it (or from another
// map of the same square, such as skin.pure).
function skinAt(skin, x, y, map = skin.data) {
  const s = skin.size;
  const u = Math.floor(((x - skin.x0) / skin.side) * s);
  const v = Math.floor(((y - skin.y0) / skin.side) * s);
  if (u < 0 || v < 0 || u >= s || v >= s) return 1;
  return smoothstep(TUNE.maskLo, Math.max(TUNE.maskLo + 1e-3, TUNE.maskHi), map[v * s + u] / 255);
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

// Your camera picture drawn through the face mesh: sampled where your face points are, drawn where the
// reshaped points are, so your real eyes and mouth move with the new face shape.
const VERTEX_WARP = `#version 300 es
in vec2 aPos;     // reshaped position in pixels
in vec2 aUV;      // where this point is in the camera picture (0..1)
uniform vec2 uViewport;
out vec2 vUV;
void main() {
  vUV = aUV;
  gl_Position = vec4(aPos.x / uViewport.x * 2.0 - 1.0, 1.0 - aPos.y / uViewport.y * 2.0, 0.0, 1.0);
}`;

const VERTEX_FACE = `#version 300 es
in vec2 aPos;     // target position in pixels
in vec2 aSrcUV;   // where this point is in the photo (0..1)
in vec2 aFrameUV; // where this point is in the camera picture (0..1), before reshaping
in float aAlpha;  // edge feather x turned-away fade
in float aEdge;   // 1 = this triangle reaches the soft edge at the face outline (worked out per pixel)
uniform vec2 uViewport;
out vec2 vSrcUV;
out vec2 vFrameUV;
out float vAlpha;
out float vEdge;
void main() {
  vSrcUV = aSrcUV;
  vFrameUV = aFrameUV;
  vAlpha = aAlpha;
  vEdge = aEdge;
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
uniform sampler2D uMask; // the photo's skin confidence, over a square around its face
uniform vec4 uMaskBox;   // that square in photo UV: corner, size
uniform vec3 uMaskSet;   // confidence where the photo starts to show, fully shows; strength
uniform vec2 uOval[${FACE_OVAL.length}]; // the face outline where it is drawn, in output pixels
uniform float uOvalW[${FACE_OVAL.length}]; // the soft edge's width at each outline point, in pixels
in vec2 vSrcUV;
in vec2 vFrameUV;
in float vAlpha;
in float vEdge;
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

// Distance inside the face outline in soft-edge widths (uOvalW), 0 on or outside it. Per pixel, because
// the outline's triangles are long and thin on a turned cheek, and an edge drawn through their corners
// alone comes out saw-toothed.
float outlineDist(vec2 p) {
  float best = 1e12;
  bool inside = false;
  const int N = ${FACE_OVAL.length};
  for (int k = 0, j = N - 1; k < N; j = k, k++) {
    vec2 a = uOval[j];
    vec2 b = uOval[k];
    if ((b.y > p.y) != (a.y > p.y) && p.x < (a.x - b.x) * (p.y - b.y) / (a.y - b.y) + b.x) inside = !inside;
    vec2 d = b - a;
    float t = clamp(dot(p - a, d) / max(dot(d, d), 1e-6), 0.0, 1.0);
    vec2 e = p - a - d * t;
    float w = mix(uOvalW[j], uOvalW[k], t);
    best = min(best, dot(e, e) / (w * w));
  }
  return inside ? sqrt(best) : 0.0;
}

void main() {
  vec2 fuv = vFrameUV; // your camera at this spot of the (reshaped) face
  vec3 src = transfer(texture(uSrc, vSrcUV).rgb);
  vec3 srcLow = transfer(textureLod(uSrc, vSrcUV, uSrcLod).rgb);
  vec3 frame = texture(uFrame, fuv).rgb;
  vec3 frameLow = textureLod(uFrame, fuv, uFrameLod).rgb;
  // Photo detail + the room's shading on your face + your live skin highlights.
  vec3 col = src + (frameLow - srcLow) * uShading + (frame - frameLow) * uDetail;
  // Only the photo's skin: where it has hair or glasses, your own (reshaped) picture shows.
  float skin = texture(uMask, (vSrcUV - uMaskBox.xy) / uMaskBox.zw).r;
  float alpha = vAlpha * mix(1.0, smoothstep(uMaskSet.x, uMaskSet.y, skin), uMaskSet.z);
  if (vEdge > 0.5) alpha *= smoothstep(0.0, 1.0, outlineDist(vec2(gl_FragCoord.x, uViewport.y - gl_FragCoord.y)));
  if (uCamGlasses > 0.0) {
    // Your glasses rims (thin dark lines in the glasses area) stay on top.
    vec2 fpx = vFrameUV * uViewport; // where this spot is on your real face
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
    this.rawZ = new Float32Array(MESH_POINTS); // this frame's tracked depths (px)
    this.depth = new Float32Array(MESH_POINTS); // smoothed depths, for the face shape morph
    this.depthFor = null;
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
    // Hair/skin segmentation of the photo (once per photo). Without it the whole face outline is drawn.
    try {
      const segModel = await this.api.readAsset('mediapipe/selfie_multiclass_256x256.tflite');
      this.segmenter = await ImageSegmenter.createFromOptions(fileset, {
        baseOptions: { modelAssetBuffer: new Uint8Array(segModel), delegate: 'CPU' },
        runningMode: 'IMAGE',
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      });
    } catch (err) {
      this.segmenter = null;
      console.warn(`Lite: no hair/skin segmentation: ${(err && err.message) || err}`);
    }
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
    this.warpProgram = program(gl, VERTEX_WARP, FRAGMENT_QUAD);
    this.warpBuffer = gl.createBuffer();
    this.warpIndex = gl.createBuffer();
    this.frameTex = makeTexture(gl);
    this.srcTex = makeTexture(gl);
    this.maskTex = makeTexture(gl);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    this.loc = {
      qPos: gl.getAttribLocation(this.quadProgram, 'aPos'),
      qFrame: gl.getUniformLocation(this.quadProgram, 'uFrame'),
      fPos: gl.getAttribLocation(this.faceProgram, 'aPos'),
      fSrcUV: gl.getAttribLocation(this.faceProgram, 'aSrcUV'),
      fFrameUV: gl.getAttribLocation(this.faceProgram, 'aFrameUV'),
      fAlpha: gl.getAttribLocation(this.faceProgram, 'aAlpha'),
      fEdge: gl.getAttribLocation(this.faceProgram, 'aEdge'),
      wPos: gl.getAttribLocation(this.warpProgram, 'aPos'),
      wUV: gl.getAttribLocation(this.warpProgram, 'aUV'),
      wFrame: gl.getUniformLocation(this.warpProgram, 'uFrame'),
      wViewport: gl.getUniformLocation(this.warpProgram, 'uViewport'),
    };
    for (const name of ['uSrc', 'uFrame', 'uViewport', 'uSrcMean', 'uSrcStd', 'uDstMean', 'uDstStd', 'uSrcLod', 'uFrameLod', 'uShading', 'uDetail', 'uStrength', 'uCamGlasses', 'uGlassEd', 'uGlassLod', 'uGlassR', 'uGlassL', 'uGlassAx', 'uMask', 'uMaskBox', 'uMaskSet', 'uOval', 'uOvalW']) {
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
      result = { faceLandmarks: [lm.map((p) => ({ x: (p.x * padded.width - pad) / width, y: (p.y * padded.height - pad) / height, z: (p.z * padded.width) / width }))] };
    }

    const lm = result.faceLandmarks[0];
    const points = new Float32Array(MESH_POINTS * 2);
    const points3 = new Float32Array(MESH_POINTS * 3); // with depth, for the face shape morph
    for (let i = 0; i < MESH_POINTS; i += 1) {
      points[i * 2] = lm[i].x * width;
      points[i * 2 + 1] = lm[i].y * height;
      points3[i * 3] = points[i * 2];
      points3[i * 3 + 1] = points[i * 2 + 1];
      points3[i * 3 + 2] = (lm[i].z || 0) * width;
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

    // Fixed topology from the photo: never re-triangulated, so the mesh can't flicker. It is built on
    // the photo's face turned to look straight on (from its 3D points), so a cheek the photo shows
    // turned away (where the outline runs inside the cheek) still gets its triangles.
    const layout = TUNE.frontalMesh ? frontalLayout(points3) : points;
    const lpt = (i) => [layout[i * 2], layout[i * 2 + 1]];
    const lOval = FACE_OVAL.map(lpt);
    const lHoles = HOLES.map((ring) => ring.map(lpt));
    // How far up the forehead each outline point is (0 at the eyes and below, 1 at the top).
    const eyeY = (layout[33 * 2 + 1] + layout[263 * 2 + 1]) / 2;
    const ovalUp = Float32Array.from(FACE_OVAL, (i) => smoothstep(0, 1, (eyeY - layout[i * 2 + 1]) / (eyeY - layout[10 * 2 + 1] || 1)));
    const del = new Delaunator(layout);
    const tris = [];
    for (let t = 0; t < del.triangles.length; t += 3) {
      const [a, b, c] = [del.triangles[t], del.triangles[t + 1], del.triangles[t + 2]];
      const cx = (layout[a * 2] + layout[b * 2] + layout[c * 2]) / 3;
      const cy = (layout[a * 2 + 1] + layout[b * 2 + 1] + layout[c * 2 + 1]) / 3;
      if (!inPolygon(cx, cy, lOval)) continue; // outside the face
      if (lHoles.some((h) => inPolygon(cx, cy, h))) continue; // your real eyes and mouth show through
      if (TUNE.spanCut && spansHole(a, b, c)) continue;
      tris.push(a, b, c);
    }

    // Soft edges: fade towards the face outline and around the eye and mouth holes.
    const faceWidth = Math.max(...oval.map((p) => p[0])) - Math.min(...oval.map((p) => p[0]));
    const feather = new Float32Array(MESH_POINTS);
    const holeFeather = new Float32Array(MESH_POINTS); // the eye and mouth holes' part alone
    for (let i = 0; i < MESH_POINTS; i += 1) {
      const [x, y] = pt(i);
      const edge = inPolygon(x, y, oval) ? distToPolygonEdge(x, y, oval) / faceWidth : 0;
      let a = 1;
      holes.forEach((h, k) => {
        const mouth = k === 2;
        const d = distToPolygonEdge(x, y, h) / faceWidth;
        const w = Math.max(1e-4, mouth ? TUNE.mouthFeather : TUNE.holeFeather);
        a = Math.min(a, mouth ? smoothstep(0.0, w, d) : TUNE.eyeEdge + (1 - TUNE.eyeEdge) * smoothstep(0.0, w, d));
      });
      // Points the photo shows beyond its own outline (a cheek turned away) may hold hair or background.
      if (TUNE.frontalMesh && !FACE_OVAL.includes(i) && !inPolygon(x, y, oval)) a *= 1 - smoothstep(0, Math.max(1e-4, TUNE.beyondFeather), distToPolygonEdge(x, y, oval) / faceWidth);
      holeFeather[i] = a;
      feather[i] = Math.min(a, smoothstep(0.0, Math.max(1e-4, TUNE.feather), edge));
    }
    const srcAreas = new Float32Array(tris.length / 3);
    for (let t = 0; t < tris.length; t += 3) srcAreas[t / 3] = triArea(points, tris[t], tris[t + 1], tris[t + 2]);
    // How squarely each part of the face looks at the camera in the photo (its 3D normal's share
    // along the view, smoothed over the mesh): a cheek seen edge-on in the photo has too few pixels
    // to cover a cheek that faces you.
    const { nbr, nbrAt } = neighbours(tris, MESH_POINTS);
    const orient = triArea(layout, tris[0], tris[1], tris[2]) < 0 ? -1 : 1; // the mesh's winding
    const pointFacing = new Float32Array(MESH_POINTS).fill(1);
    {
      const z = new Float32Array(MESH_POINTS);
      for (let i = 0; i < MESH_POINTS; i += 1) z[i] = points3[i * 3 + 2];
      const nrm = new Float32Array(MESH_POINTS * 3);
      for (let t = 0; t < tris.length; t += 3) addNormal(nrm, points, z, tris[t], tris[t + 1], tris[t + 2], orient);
      smoothVectors(nrm, new Float32Array(MESH_POINTS * 3), nbr, nbrAt, TUNE.edgeOnSmooth);
      for (let i = 0; i < MESH_POINTS; i += 1) {
        const len = Math.hypot(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]);
        if (len > 0) pointFacing[i] = nrm[i * 3 + 2] / len;
      }
    }

    // Face shape: which points are lips (they move as one piece), and a mesh for warping your camera
    // picture: every face point (eyes and mouth too) plus a fixed ring outside the outline, so the
    // reshape stretches the hair and neck smoothly instead of tearing at the edge of the face.
    // lips[i]: 0 = not a lip point, else 1 + where it sits across the lip (0 = mouth opening, 1 = outer edge).
    const lips = new Float32Array(MESH_POINTS);
    const lipRing = OUTER_LIPS.map(pt);
    const mouthRing = INNER_LIPS.map(pt);
    for (let i = 0; i < MESH_POINTS; i += 1) {
      const [x, y] = pt(i);
      if (!OUTER_LIPS.includes(i) && !inPolygon(x, y, lipRing)) continue;
      const toMouth = INNER_LIPS.includes(i) ? 0 : distToPolygonEdge(x, y, mouthRing);
      const toEdge = OUTER_LIPS.includes(i) ? 0 : distToPolygonEdge(x, y, lipRing);
      lips[i] = 1 + toMouth / (toMouth + toEdge || 1);
    }
    // nearMouth[i]: distance from the lips as a share of the face width (smiles move the skin around them).
    const nearMouth = new Float32Array(MESH_POINTS);
    for (let i = 0; i < MESH_POINTS; i += 1) if (!lips[i]) nearMouth[i] = distToPolygonEdge(...pt(i), lipRing) / faceWidth;
    // jaw[i]: how far each point goes down when the mouth opens (lower lip and chin 1, upper lip and above 0).
    const jaw = new Float32Array(MESH_POINTS);
    const downX = points[2 * 2] - points[168 * 2];
    const downY = points[2 * 2 + 1] - points[168 * 2 + 1];
    const level = (i) => (points[i * 2] - points[168 * 2]) * downX + (points[i * 2 + 1] - points[168 * 2 + 1]) * downY;
    const cornerLevel = (level(61) + level(291)) / 2;
    for (let i = 0; i < MESH_POINTS; i += 1) {
      if (lips[i]) jaw[i] = LOWER_LIP.includes(i) ? 1 : MOUTH_CORNERS.includes(i) ? 0.5 : 0;
      else jaw[i] = smoothstep(cornerLevel, level(17), level(i));
    }
    const warpPts = new Float32Array((MESH_POINTS + FACE_OVAL.length) * 2);
    warpPts.set(points);
    ringAround(points, warpPts, MESH_POINTS * 2);
    const warpTris = Uint16Array.from(new Delaunator(warpPts).triangles);
    // Each face point's neighbours in that mesh (to smooth out folds): warpNbr[warpNbrAt[i] .. warpNbrAt[i + 1]).
    const nbrSets = Array.from({ length: MESH_POINTS }, () => new Set());
    for (let t = 0; t < warpTris.length; t += 3) {
      for (let k = 0; k < 3; k += 1) {
        const a = warpTris[t + k];
        if (a < MESH_POINTS) {
          nbrSets[a].add(warpTris[t + ((k + 1) % 3)]);
          nbrSets[a].add(warpTris[t + ((k + 2) % 3)]);
        }
      }
    }
    const warpNbrAt = new Int32Array(MESH_POINTS + 1);
    for (let i = 0; i < MESH_POINTS; i += 1) warpNbrAt[i + 1] = warpNbrAt[i] + nbrSets[i].size;
    const warpNbr = Int32Array.from(nbrSets.flatMap((s) => [...s]));

    // The photo's skin (hair, glasses and background inside the outline are left out). If hardly any of
    // the face counts as skin, the segmentation has failed (a drawing, an odd crop): draw it all.
    let skin = this.segmenter && TUNE.mask > 0 ? segmentSkin(this.segmenter, canvas, oval) : null;
    if (skin && TUNE.maskPaintGlasses > 0) paintOut(ctx, width, height, skin);
    const pixels = ctx.getImageData(0, 0, width, height).data;
    if (skin) {
      let sum = 0;
      let n = 0;
      for (let y = 0; y < height; y += 4) {
        for (let x = 0; x < width; x += 4) {
          if (!inPolygon(x + 0.5, y + 0.5, oval)) continue;
          sum += skinAt(skin, x + 0.5, y + 0.5);
          n += 1;
        }
      }
      skin.coverage = n ? sum / n : 0;
      if (skin.coverage < SKIN_MIN_COVERAGE) skin = null;
    }
    const weight = skin && TUNE.maskStats ? (x, y) => skinAt(skin, x, y, skin.pure) : null;
    const stats = (weight && labStats(pixels, width, height, oval, weight)) ||
      labStats(pixels, width, height, oval) || {
        mean: [60, 10, 15],
        std: [15, 5, 7],
      };

    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, canvas);
    gl.generateMipmap(gl.TEXTURE_2D);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    if (skin) gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, skin.size, skin.size, 0, gl.RED, gl.UNSIGNED_BYTE, skin.data);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([255]));
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    const maskBox = skin ? [skin.x0 / width, skin.y0 / height, skin.side / width, skin.side / height] : [0, 0, 1, 1];
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.warpIndex);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, warpTris, gl.STATIC_DRAW);

    this.smoother.reset();
    this.dstStats = null;
    return { width, height, points, points3, lips, nearMouth, jaw, warpTris, warpNbr, warpNbrAt, warpCount: warpTris.length, tris: Uint16Array.from(tris), feather, holeFeather, ovalUp, srcAreas, orient, nbr, nbrAt, pointFacing, stats, faceWidth, masked: Boolean(skin), maskBox, skinCoverage: skin ? skin.coverage : null };
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
        this.rawZ[i] = (landmarks[i].z || 0) * output.width;
      }
    }
    // While tracking is lost for a moment, the last good points stay (fading out).
    const points = this.smoother.update(landmarks ? raw : null, t);
    // Depth only aligns the photo's shape to your head in 3D; it's kept apart from the 2D smoother
    // (which fits the head move on x,y pairs) and lightly smoothed, restarting with the smoother.
    if (points && landmarks) {
      const restart = this.depthFor !== points;
      this.depthFor = points;
      const k = restart ? 1 : TUNE.depthSmooth;
      for (let i = 0; i < MESH_POINTS; i += 1) this.depth[i] += (this.rawZ[i] - this.depth[i]) * k;
    }
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

    // Your face reshaped to the photo's proportions: your camera picture (real eyes and mouth included)
    // is warped to the new shape first, so the photo's face below lines up with it.
    const src = this.source;
    let shaped = TUNE.shape > 0 ? this.reshape(points) : points;
    if (shaped !== points && strength < 1) {
      // While the face fades in or out, the new shape eases in and out with it (no popping).
      const eased = this.eased || (this.eased = new Float32Array(points.length));
      for (let i = 0; i < points.length; i += 1) eased[i] = points[i] + (shaped[i] - points[i]) * strength;
      shaped = eased;
    }
    if (shaped !== points) this.drawWarp(points, shaped);

    // Then the photo's face, warped onto yours.
    const tris = src.tris;
    const triCount = tris.length / 3;
    let minX = Infinity;
    let maxX = -Infinity;
    let minSX = Infinity;
    let maxSX = -Infinity;
    for (let k = 0; k < FACE_OVAL.length; k += 1) {
      const x = points[FACE_OVAL[k] * 2];
      const sx = shaped[FACE_OVAL[k] * 2];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (sx < minSX) minSX = sx;
      if (sx > maxSX) maxSX = sx;
    }
    const targetWidth = maxX - minX;
    const scale2 = ((maxSX - minSX) / src.faceWidth) ** 2;
    // One vertex array per photo, reused every frame: x, y (reshaped), photo u, v (fixed),
    // camera u, v (where that spot of your face is in the camera picture), alpha.
    if (!src.verts) {
      src.verts = new Float32Array(triCount * 3 * 8);
      for (let v = 0; v < triCount * 3; v += 1) {
        src.verts[v * 8 + 2] = src.points[tris[v] * 2] / src.width;
        src.verts[v * 8 + 3] = src.points[tris[v] * 2 + 1] / src.height;
      }
    }
    const verts = src.verts;
    const iw = 1 / output.width;
    const ih = 1 / output.height;
    // Parts of the face turned away fade out, measured per point and smoothed over the mesh so that
    // neighbouring triangles' opacity meets without cracks: where your (reshaped) face is much smaller
    // than the photo's (turning away from the camera), much larger (the photo's own side is turned
    // away, too few pixels to cover it), or seen edge-on in the camera or in the photo (3D normals).
    // Single triangles that flip or collapse (folds) fade on their own.
    const triFade = this.triFade && this.triFade.length === triCount ? this.triFade : (this.triFade = new Float32Array(triCount));
    const pointFade = this.pointFade || (this.pointFade = new Float32Array(MESH_POINTS));
    pointFade.fill(1);
    const pf = this.fadeWork || (this.fadeWork = { n: new Float32Array(MESH_POINTS * 3), m: new Float32Array(MESH_POINTS * 3), lr: new Float32Array(MESH_POINTS), w: new Float32Array(MESH_POINTS), tmp: new Float32Array(MESH_POINTS) });
    const smoothTurn = TUNE.turnSmooth > 0;
    const useEdge = TUNE.edgeOnHi > 0;
    const useRatio = smoothTurn || TUNE.stretchHi > 0;
    if (useEdge) pf.n.fill(0);
    if (useRatio) {
      pf.lr.fill(0);
      pf.w.fill(0);
    }
    for (let t = 0; t < triCount; t += 1) {
      const a = tris[t * 3];
      const b = tris[t * 3 + 1];
      const c = tris[t * 3 + 2];
      const ratio = triArea(shaped, a, b, c) / (src.srcAreas[t] * scale2 || 1);
      const fade = smoothTurn ? smoothstep(TUNE.flipLo, TUNE.flipHi, ratio) : smoothstep(TUNE.turnLo, TUNE.turnHi, ratio);
      triFade[t] = fade;
      if (useEdge) addNormal(pf.n, points, this.depth, a, b, c, src.orient);
      if (useRatio) {
        // Area-weighted log of how much the photo's triangle is stretched (> 0) or squeezed (< 0).
        const w = Math.abs(src.srcAreas[t]);
        const lr = Math.log(Math.max(0.02, ratio)) * w;
        pf.lr[a] += lr;
        pf.lr[b] += lr;
        pf.lr[c] += lr;
        pf.w[a] += w;
        pf.w[b] += w;
        pf.w[c] += w;
      }
    }
    if (useRatio) {
      for (let i = 0; i < MESH_POINTS; i += 1) pf.lr[i] = pf.w[i] > 0 ? pf.lr[i] / pf.w[i] : 0;
      smoothScalars(pf.lr, pf.tmp, src.nbr, src.nbrAt, TUNE.turnSmooth);
      const turnLo = Math.log(TUNE.turnLo);
      const turnHi = Math.log(TUNE.turnHi);
      const stretchLo = Math.log(TUNE.stretchLo);
      const stretchHi = Math.log(TUNE.stretchHi);
      for (let i = 0; i < MESH_POINTS; i += 1) {
        if (smoothTurn) pointFade[i] *= smoothstep(turnLo, turnHi, pf.lr[i]);
        if (TUNE.stretchHi > 0) pointFade[i] *= 1 - smoothstep(stretchLo, stretchHi, pf.lr[i]);
      }
    }
    if (useEdge) {
      smoothVectors(pf.n, pf.m, src.nbr, src.nbrAt, TUNE.edgeOnSmooth);
      for (let i = 0; i < MESH_POINTS; i += 1) {
        const nz = pf.n[i * 3 + 2];
        const len = Math.hypot(pf.n[i * 3], pf.n[i * 3 + 1], nz) || 1;
        pointFade[i] *= smoothstep(TUNE.edgeOnLo, TUNE.edgeOnHi, Math.min(nz / len, src.pointFacing[i]));
      }
    }
    // The soft edge at the face outline, measured where the face is drawn (the photo's own outline
    // is narrow on a side it shows turned away, and that would turn into a wide, jagged edge here).
    const live = TUNE.featherLive ? this.liveFeather || (this.liveFeather = new Float32Array(MESH_POINTS)) : null;
    if (live) outlineFeather(shaped, live, (maxSX - minSX) * Math.max(TUNE.feather, TUNE.featherTop));
    for (let t = 0; t < triCount; t += 1) {
      const o = t * 24;
      const edge = live && (live[tris[t * 3]] < 1 || live[tris[t * 3 + 1]] < 1 || live[tris[t * 3 + 2]] < 1) ? 1 : 0;
      for (let k = 0; k < 3; k += 1) {
        const i = tris[t * 3 + k];
        const q = o + k * 8;
        verts[q] = shaped[i * 2];
        verts[q + 1] = shaped[i * 2 + 1];
        verts[q + 4] = points[i * 2] * iw;
        verts[q + 5] = points[i * 2 + 1] * ih;
        verts[q + 6] = (live ? src.holeFeather[i] : src.feather[i]) * triFade[t] * pointFade[i];
        verts[q + 7] = edge;
      }
    }

    gl.useProgram(this.faceProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(this.loc.uSrc, 0);
    gl.uniform1i(this.loc.uFrame, 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.maskTex);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(this.loc.uMask, 2);
    gl.uniform4fv(this.loc.uMaskBox, src.maskBox);
    gl.uniform3f(this.loc.uMaskSet, TUNE.maskLo, Math.max(TUNE.maskLo + 1e-3, TUNE.maskHi), src.masked ? Math.min(1, TUNE.mask) : 0);
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
    if (live) {
      const oval = this.ovalPx || (this.ovalPx = new Float32Array(FACE_OVAL.length * 2));
      for (let k = 0; k < FACE_OVAL.length; k += 1) {
        oval[k * 2] = shaped[FACE_OVAL[k] * 2];
        oval[k * 2 + 1] = shaped[FACE_OVAL[k] * 2 + 1];
      }
      gl.uniform2fv(this.loc.uOval, oval);
      // A thin edge along the cheeks and jaw, a wider one up the forehead (towards the hairline).
      const ovalW = this.ovalW || (this.ovalW = new Float32Array(FACE_OVAL.length));
      const fw = maxSX - minSX;
      for (let k = 0; k < FACE_OVAL.length; k += 1) ovalW[k] = Math.max(1, fw * (TUNE.feather + (TUNE.featherTop - TUNE.feather) * src.ovalUp[k]));
      gl.uniform1fv(this.loc.uOvalW, ovalW);
    }
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
    const stride = 8 * 4;
    gl.enableVertexAttribArray(this.loc.fPos);
    gl.vertexAttribPointer(this.loc.fPos, 2, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(this.loc.fSrcUV);
    gl.vertexAttribPointer(this.loc.fSrcUV, 2, gl.FLOAT, false, stride, 8);
    gl.enableVertexAttribArray(this.loc.fFrameUV);
    gl.vertexAttribPointer(this.loc.fFrameUV, 2, gl.FLOAT, false, stride, 16);
    gl.enableVertexAttribArray(this.loc.fAlpha);
    gl.vertexAttribPointer(this.loc.fAlpha, 1, gl.FLOAT, false, stride, 24);
    gl.enableVertexAttribArray(this.loc.fEdge);
    gl.vertexAttribPointer(this.loc.fEdge, 1, gl.FLOAT, false, stride, 28);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.TRIANGLES, 0, triCount * 3);
    gl.disable(gl.BLEND);
    gl.disableVertexAttribArray(this.loc.fPos);
    gl.disableVertexAttribArray(this.loc.fSrcUV);
    gl.disableVertexAttribArray(this.loc.fFrameUV);
    gl.disableVertexAttribArray(this.loc.fAlpha);
    gl.disableVertexAttribArray(this.loc.fEdge);
  }

  // Your face points moved towards the photo's face shape. The photo's points are first aligned to
  // your head (rotation, scale and shift fitted on points that don't move with expression), so only
  // proportions change: head pose stays yours. Lips mostly move as one piece and your eyes keep their
  // own shape (only moved and scaled to the photo's eye corners), so blinks and speech come through.
  reshape(points) {
    const src = this.source;
    const n = MESH_POINTS;
    const cam = this.cam3 || (this.cam3 = new Float32Array(n * 3));
    const flat = !TUNE.shape3d;
    if (flat && !src.points2) src.points2 = src.points3.map((v, i) => (i % 3 === 2 ? 0 : v));
    const photo = flat ? src.points2 : src.points3;
    for (let i = 0; i < n; i += 1) {
      cam[i * 3] = points[i * 2];
      cam[i * 3 + 1] = points[i * 2 + 1];
      cam[i * 3 + 2] = flat ? 0 : this.depth[i];
    }
    const { s, r, ca, cb } = fitSimilarity3(photo, cam, SHAPE_FIT);
    // Your resting face: a slow average of your points with the head pose taken out (in the photo's
    // frame). The move is photo minus rest, so expressions (you now minus you at rest) pass straight
    // through: an open mouth, raised brows or a blink are not pulled back to the photo's.
    // A restarted smoother hands over a new points array, and that restarts the rest shape too.
    const now = performance.now();
    const fresh = !this.rest || this.restFor !== points || this.restFlat !== flat;
    if (fresh) {
      this.rest = new Float32Array(n * 3);
      this.restFor = points;
      this.restFlat = flat;
      this.restFrames = 0;
    }
    const rest = this.rest;
    // A plain running mean at first (no single start-up frame dominates), then a slow moving average.
    this.restFrames += 1;
    const slow = TUNE.shapeRest > 0 && !fresh ? 1 - Math.exp(-Math.max(0, now - this.restAt) / 1000 / TUNE.shapeRest) : 1;
    const follow = Math.max(1 / this.restFrames, slow);
    this.restAt = now;
    for (let i = 0; i < n; i += 1) {
      const x = cam[i * 3] - cb[0];
      const y = cam[i * 3 + 1] - cb[1];
      const z = cam[i * 3 + 2] - cb[2];
      for (let k = 0; k < 3; k += 1) {
        const q = (r[k] * x + r[3 + k] * y + r[6 + k] * z) / s + ca[k]; // your point in the photo's frame
        rest[i * 3 + k] += (q - rest[i * 3 + k]) * follow;
      }
    }
    // The photo's own expression shouldn't come over either: as far as its mouth is open wider than
    // yours at rest, lift its lower lip, chin and lower face back up (along the nose, the face's "down").
    let ux = photo[2 * 3] - photo[168 * 3];
    let uy = photo[2 * 3 + 1] - photo[168 * 3 + 1];
    let uz = photo[2 * 3 + 2] - photo[168 * 3 + 2];
    const ul = Math.hypot(ux, uy, uz) || 1;
    ux /= ul;
    uy /= ul;
    uz /= ul;
    const gap = (p, i, j) => (p[j * 3] - p[i * 3]) * ux + (p[j * 3 + 1] - p[i * 3 + 1]) * uy + (p[j * 3 + 2] - p[i * 3 + 2]) * uz;
    const open = (gap(photo, 13, 14) - gap(rest, 13, 14)) * TUNE.shapeJaw;
    // d: from each of your points to where it goes (the aligned photo point, when you are at rest).
    const d = this.shapeD || (this.shapeD = new Float32Array(n * 2));
    for (let i = 0; i < n; i += 1) {
      const lift = src.jaw[i] * open;
      const ox = photo[i * 3] - rest[i * 3] - ux * lift;
      const oy = photo[i * 3 + 1] - rest[i * 3 + 1] - uy * lift;
      const oz = photo[i * 3 + 2] - rest[i * 3 + 2] - uz * lift;
      d[i * 2] = s * (r[0] * ox + r[1] * oy + r[2] * oz);
      d[i * 2 + 1] = s * (r[3] * ox + r[4] * oy + r[5] * oz);
    }
    // Mouth: the upper lip centre's move (it rides on the skull) for the lips and the skin around them,
    // plus only part of their own shape: the photo's smile (corners, cheeks) is expression, not shape.
    // Further than shapeZone (share of the face width) from the lips, points take their full shape.
    const lx = (d[0] + d[37 * 2] + d[267 * 2]) / 3;
    const ly = (d[1] + d[37 * 2 + 1] + d[267 * 2 + 1]) / 3;
    for (let i = 0; i < n; i += 1) {
      let w;
      if (src.lips[i]) w = TUNE.shapeMouth + (TUNE.shapeLips - TUNE.shapeMouth) * (src.lips[i] - 1);
      else if (src.nearMouth[i] < TUNE.shapeZone) w = TUNE.shapeLips + (1 - TUNE.shapeLips) * smoothstep(0, TUNE.shapeZone, src.nearMouth[i]);
      else continue;
      d[i * 2] = lx + (d[i * 2] - lx) * w;
      d[i * 2 + 1] = ly + (d[i * 2 + 1] - ly) * w;
    }
    // Eyes: the similarity that takes your eye corners to the photo's, applied to the whole eye.
    [RIGHT_EYE, LEFT_EYE].forEach((ring, e) => {
      const [a, b] = EYE_CORNERS[e];
      const vx = points[b * 2] - points[a * 2];
      const vy = points[b * 2 + 1] - points[a * 2 + 1];
      const wx = vx + d[b * 2] - d[a * 2];
      const wy = vy + d[b * 2 + 1] - d[a * 2 + 1];
      const len = vx * vx + vy * vy || 1;
      const kr = (wx * vx + wy * vy) / len - 1; // (corner-to-corner change) − 1, as a complex number
      const ki = (wy * vx - wx * vy) / len;
      const ax = d[a * 2];
      const ay = d[a * 2 + 1];
      for (const i of ring) {
        if (i === a || i === b) continue;
        const px = points[i * 2] - points[a * 2];
        const py = points[i * 2 + 1] - points[a * 2 + 1];
        const rx = ax + kr * px - ki * py;
        const ry = ay + ki * px + kr * py;
        d[i * 2] = rx + (d[i * 2] - rx) * TUNE.shapeEyes;
        d[i * 2 + 1] = ry + (d[i * 2 + 1] - ry) * TUNE.shapeEyes;
      }
    });
    // No folds: where the reshape would turn a triangle of the camera warp over, smooth its corners'
    // moves towards their neighbours' (and, if that isn't enough, shrink them), over a few rounds.
    // A fold shows as torn, doubled skin (at the jaw and cheeks when the photo's head is turned
    // differently, the nostrils and the mouth corners).
    // from/to: warp points (face points, then the fixed ring) before and after the reshape.
    const m = n + FACE_OVAL.length;
    const from = this.foldFrom || (this.foldFrom = new Float32Array(m * 2));
    const to = this.foldTo || (this.foldTo = new Float32Array(m * 2));
    from.set(points.subarray(0, n * 2));
    ringAround(points, from, n * 2);
    for (let i = 0; i < n * 2; i += 1) to[i] = points[i] + d[i] * TUNE.shape;
    to.set(from.subarray(n * 2), n * 2);
    const tri = src.warpTris;
    const triCount = tri.length / 3;
    if (!this.foldBefore || this.foldBefore.length !== triCount) this.foldBefore = new Float32Array(triCount);
    const before = this.foldBefore;
    const eased = this.foldEased || (this.foldEased = new Int32Array(m));
    eased.fill(0);
    const { warpNbr, warpNbrAt } = src;
    const area = (p, t) => {
      const a = tri[t] * 2;
      const b = tri[t + 1] * 2;
      const c = tri[t + 2] * 2;
      return (p[b] - p[a]) * (p[c + 1] - p[a + 1]) - (p[c] - p[a]) * (p[b + 1] - p[a + 1]);
    };
    for (let t = 0; t < triCount; t += 1) before[t] = area(from, t * 3);
    for (let round = 1; round <= FOLD_ROUNDS; round += 1) {
      let any = false;
      for (let t = 0; t < triCount; t += 1) {
        const a = tri[t * 3];
        const b = tri[t * 3 + 1];
        const c = tri[t * 3 + 2];
        // After the first round only triangles with a corner that just moved can have changed.
        if (round > 1 && eased[a] !== round - 1 && eased[b] !== round - 1 && eased[c] !== round - 1) continue;
        if (Math.abs(before[t]) < 2 || area(to, t * 3) / before[t] > 0) continue;
        for (const v of [a, b, c]) {
          if (v >= n || eased[v] === round) continue;
          eased[v] = round;
          if (round <= FOLD_SMOOTH_ROUNDS) {
            // Halfway to the neighbours' average move (the ring's points don't move).
            let sx = 0;
            let sy = 0;
            for (let k = warpNbrAt[v]; k < warpNbrAt[v + 1]; k += 1) {
              const u = warpNbr[k];
              if (u < n) {
                sx += d[u * 2];
                sy += d[u * 2 + 1];
              }
            }
            const cnt = warpNbrAt[v + 1] - warpNbrAt[v] || 1;
            d[v * 2] = (d[v * 2] + sx / cnt) / 2;
            d[v * 2 + 1] = (d[v * 2 + 1] + sy / cnt) / 2;
          } else {
            d[v * 2] *= FOLD_EASE;
            d[v * 2 + 1] *= FOLD_EASE;
          }
          to[v * 2] = points[v * 2] + d[v * 2] * TUNE.shape;
          to[v * 2 + 1] = points[v * 2 + 1] + d[v * 2 + 1] * TUNE.shape;
          any = true;
        }
      }
      if (!any) break;
    }
    return to.subarray(0, n * 2);
  }

  // Your camera picture through the whole face mesh (eyes and mouth included): sampled at your face
  // points, drawn at the reshaped ones. A ring outside the face stays put, so hair and neck just stretch.
  drawWarp(points, shaped) {
    const gl = this.gl;
    const n = MESH_POINTS;
    const m = n + FACE_OVAL.length;
    const v = this.warpVerts || (this.warpVerts = new Float32Array(m * 4));
    const ring = this.warpRing || (this.warpRing = new Float32Array(FACE_OVAL.length * 2));
    ringAround(points, ring, 0);
    for (let i = 0; i < m; i += 1) {
      const x = i < n ? points[i * 2] : ring[(i - n) * 2];
      const y = i < n ? points[i * 2 + 1] : ring[(i - n) * 2 + 1];
      v[i * 4] = i < n ? shaped[i * 2] : x;
      v[i * 4 + 1] = i < n ? shaped[i * 2 + 1] : y;
      v[i * 4 + 2] = x / output.width;
      v[i * 4 + 3] = y / output.height;
    }
    gl.useProgram(this.warpProgram);
    gl.uniform1i(this.loc.wFrame, 1);
    gl.uniform2f(this.loc.wViewport, output.width, output.height);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.warpBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(this.loc.wPos);
    gl.vertexAttribPointer(this.loc.wPos, 2, gl.FLOAT, false, 16, 0);
    gl.enableVertexAttribArray(this.loc.wUV);
    gl.vertexAttribPointer(this.loc.wUV, 2, gl.FLOAT, false, 16, 8);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.warpIndex);
    gl.drawElements(gl.TRIANGLES, this.source.warpCount, gl.UNSIGNED_SHORT, 0);
    gl.disableVertexAttribArray(this.loc.wPos);
    gl.disableVertexAttribArray(this.loc.wUV);
  }
}

// Signed area: negative when the triangle has flipped over.
function triArea(p, a, b, c) {
  return ((p[b * 2] - p[a * 2]) * (p[c * 2 + 1] - p[a * 2 + 1]) - (p[c * 2] - p[a * 2]) * (p[b * 2 + 1] - p[a * 2 + 1])) / 2;
}

// Adds a triangle's 3D normal (area-weighted, z pointing at the viewer when sgn matches its winding)
// to its three corners' sums in out (xyz per point). p: x, y pairs (px); z: depths (px).
function addNormal(out, p, z, a, b, c, sgn) {
  const ux = p[b * 2] - p[a * 2];
  const uy = p[b * 2 + 1] - p[a * 2 + 1];
  const uz = z[b] - z[a];
  const vx = p[c * 2] - p[a * 2];
  const vy = p[c * 2 + 1] - p[a * 2 + 1];
  const vz = z[c] - z[a];
  const nx = (uy * vz - uz * vy) * sgn;
  const ny = (uz * vx - ux * vz) * sgn;
  const nz = (ux * vy - uy * vx) * sgn;
  for (const v of [a, b, c]) {
    out[v * 3] += nx;
    out[v * 3 + 1] += ny;
    out[v * 3 + 2] += nz;
  }
}

// The face's points (x, y, z per point) with the head's turn to the side undone, as x, y pairs: the
// sideways axis runs through the outer eye corners, the other stays the picture's up-down (a nod or
// a tilt doesn't change which triangles a mesh needs; a turn does).
function frontalLayout(p3) {
  const norm = (a) => {
    const l = Math.hypot(a[0], a[1], a[2]) || 1;
    return [a[0] / l, a[1] / l, a[2] / l];
  };
  const ex = norm([p3[263 * 3] - p3[33 * 3], p3[263 * 3 + 1] - p3[33 * 3 + 1], p3[263 * 3 + 2] - p3[33 * 3 + 2]]);
  const ey = norm([-ex[1] * ex[0], 1 - ex[1] * ex[1], -ex[1] * ex[2]]);
  const out = new Float32Array(MESH_POINTS * 2);
  for (let i = 0; i < MESH_POINTS; i += 1) {
    const x = p3[i * 3];
    const y = p3[i * 3 + 1];
    const z = p3[i * 3 + 2];
    out[i * 2] = x * ex[0] + y * ex[1] + z * ex[2];
    out[i * 2 + 1] = x * ey[0] + y * ey[1] + z * ey[2];
  }
  return out;
}

// Soft edge at the face outline for every face point: 0 on (or outside) the outline of p, rising to
// 1 at `width` pixels inside it.
function outlineFeather(p, out, width) {
  const m = FACE_OVAL.length;
  const w = Math.max(1e-3, width);
  for (let i = 0; i < MESH_POINTS; i += 1) {
    const x = p[i * 2];
    const y = p[i * 2 + 1];
    let inside = false;
    let best = Infinity;
    for (let k = 0, j = m - 1; k < m; j = k, k += 1) {
      const x1 = p[FACE_OVAL[j] * 2];
      const y1 = p[FACE_OVAL[j] * 2 + 1];
      const x2 = p[FACE_OVAL[k] * 2];
      const y2 = p[FACE_OVAL[k] * 2 + 1];
      if (y2 > y !== y1 > y && x < ((x1 - x2) * (y - y2)) / (y1 - y2) + x2) inside = !inside;
      const dx = x2 - x1;
      const dy = y2 - y1;
      const t = Math.max(0, Math.min(1, ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy || 1)));
      const ex = x - (x1 + t * dx);
      const ey = y - (y1 + t * dy);
      const d = ex * ex + ey * ey;
      if (d < best) best = d;
    }
    out[i] = inside ? smoothstep(0, w, Math.sqrt(best)) : 0;
  }
}

// `rounds` rounds of moving each point's value halfway to its neighbours' mean (nbr[nbrAt[i]..nbrAt[i+1])).
function smoothScalars(v, tmp, nbr, nbrAt, rounds) {
  const n = nbrAt.length - 1;
  for (let r = 0; r < rounds; r += 1) {
    for (let i = 0; i < n; i += 1) {
      const k0 = nbrAt[i];
      const k1 = nbrAt[i + 1];
      if (k1 === k0) {
        tmp[i] = v[i];
        continue;
      }
      let s = 0;
      for (let k = k0; k < k1; k += 1) s += v[nbr[k]];
      tmp[i] = (v[i] + s / (k1 - k0)) / 2;
    }
    v.set(tmp);
  }
}

function smoothVectors(v, tmp, nbr, nbrAt, rounds) {
  const n = nbrAt.length - 1;
  for (let r = 0; r < rounds; r += 1) {
    for (let i = 0; i < n; i += 1) {
      const k0 = nbrAt[i];
      const k1 = nbrAt[i + 1];
      for (let c = 0; c < 3; c += 1) {
        let s = 0;
        for (let k = k0; k < k1; k += 1) s += v[nbr[k] * 3 + c];
        tmp[i * 3 + c] = k1 > k0 ? (v[i * 3 + c] + s / (k1 - k0)) / 2 : v[i * 3 + c];
      }
    }
    v.set(tmp);
  }
}

// Each point's neighbours in a triangle list, as nbr[nbrAt[i] .. nbrAt[i + 1]).
function neighbours(tris, n) {
  const sets = Array.from({ length: n }, () => new Set());
  for (let t = 0; t < tris.length; t += 3) {
    for (let k = 0; k < 3; k += 1) {
      sets[tris[t + k]].add(tris[t + ((k + 1) % 3)]);
      sets[tris[t + k]].add(tris[t + ((k + 2) % 3)]);
    }
  }
  const nbrAt = new Int32Array(n + 1);
  for (let i = 0; i < n; i += 1) nbrAt[i + 1] = nbrAt[i] + sets[i].size;
  return { nbr: Int32Array.from(sets.flatMap((s) => [...s])), nbrAt };
}

// The face outline pushed outwards from its centre by TUNE.shapeRing, written into out from offset o.
function ringAround(points, out, o) {
  let cx = 0;
  let cy = 0;
  for (const i of FACE_OVAL) {
    cx += points[i * 2];
    cy += points[i * 2 + 1];
  }
  cx /= FACE_OVAL.length;
  cy /= FACE_OVAL.length;
  const k = 1 + TUNE.shapeRing;
  for (const i of FACE_OVAL) {
    out[o++] = cx + (points[i * 2] - cx) * k;
    out[o++] = cy + (points[i * 2 + 1] - cy) * k;
  }
}

// Eigenvector of the largest eigenvalue of a symmetric 4x4 matrix (Jacobi rotations; m is overwritten).
function topEigenvector4(m) {
  const v = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  for (let sweep = 0; sweep < 10; sweep += 1) {
    let off = 0;
    let diag = 0;
    for (let p = 0; p < 4; p += 1) {
      diag += m[p * 5] * m[p * 5];
      for (let q = p + 1; q < 4; q += 1) off += m[p * 4 + q] * m[p * 4 + q];
    }
    if (off <= 1e-24 * diag) break;
    for (let p = 0; p < 3; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        const apq = m[p * 4 + q];
        if (apq === 0) continue;
        const theta = (m[q * 5] - m[p * 5]) / (2 * apq);
        const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const kp = m[k * 4 + p];
          const kq = m[k * 4 + q];
          m[k * 4 + p] = c * kp - s * kq;
          m[k * 4 + q] = s * kp + c * kq;
        }
        for (let k = 0; k < 4; k += 1) {
          const pk = m[p * 4 + k];
          const qk = m[q * 4 + k];
          m[p * 4 + k] = c * pk - s * qk;
          m[q * 4 + k] = s * pk + c * qk;
          const vp = v[k * 4 + p];
          const vq = v[k * 4 + q];
          v[k * 4 + p] = c * vp - s * vq;
          v[k * 4 + q] = s * vp + c * vq;
        }
      }
    }
  }
  let best = 0;
  for (let i = 1; i < 4; i += 1) if (m[i * 5] > m[best * 5]) best = i;
  return [v[best], v[4 + best], v[8 + best], v[12 + best]];
}

// Least-squares similarity (scale s, rotation r, shift) taking xyz points a onto xyz points b over idx:
// b ≈ s·r·(a − ca) + cb. Horn's quaternion method, so a turned head is a rotation, not a change of shape.
function fitSimilarity3(a, b, idx) {
  const ca = [0, 0, 0];
  const cb = [0, 0, 0];
  for (const i of idx) {
    for (let k = 0; k < 3; k += 1) {
      ca[k] += a[i * 3 + k];
      cb[k] += b[i * 3 + k];
    }
  }
  for (let k = 0; k < 3; k += 1) {
    ca[k] /= idx.length;
    cb[k] /= idx.length;
  }
  const S = [0, 0, 0, 0, 0, 0, 0, 0, 0]; // S[j*3+k] = Σ a_j·b_k (centred)
  let aa = 0;
  for (const i of idx) {
    const ax = a[i * 3] - ca[0];
    const ay = a[i * 3 + 1] - ca[1];
    const az = a[i * 3 + 2] - ca[2];
    const bx = b[i * 3] - cb[0];
    const by = b[i * 3 + 1] - cb[1];
    const bz = b[i * 3 + 2] - cb[2];
    S[0] += ax * bx; S[1] += ax * by; S[2] += ax * bz;
    S[3] += ay * bx; S[4] += ay * by; S[5] += ay * bz;
    S[6] += az * bx; S[7] += az * by; S[8] += az * bz;
    aa += ax * ax + ay * ay + az * az;
  }
  const [xx, xy, xz, yx, yy, yz, zx, zy, zz] = S;
  const [w, x, y, z] = topEigenvector4([
    xx + yy + zz, yz - zy, zx - xz, xy - yx,
    yz - zy, xx - yy - zz, xy + yx, zx + xz,
    zx - xz, xy + yx, -xx + yy - zz, yz + zy,
    xy - yx, zx + xz, yz + zy, -xx - yy + zz,
  ]);
  const r = [
    w * w + x * x - y * y - z * z, 2 * (x * y - w * z), 2 * (x * z + w * y),
    2 * (x * y + w * z), w * w - x * x + y * y - z * z, 2 * (y * z - w * x),
    2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z,
  ];
  let ab = 0; // Σ b·(r a), for the scale
  for (let j = 0; j < 3; j += 1) for (let k = 0; k < 3; k += 1) ab += r[k * 3 + j] * S[j * 3 + k];
  return { s: aa > 0 ? ab / aa : 1, r, ca, cb };
}
