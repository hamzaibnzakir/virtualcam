// Face tracker points shared by the engines (MediaPipe Face Landmarker, 478 points).

export const MESH_POINT_COUNT = 468; // the face; 468..477 (irises) aren't needed

// Where the face swap's 5 alignment points (eyes, nose tip, mouth corners) sit in the tracker's
// mesh: each eye's centre (mean of its corners and lid middles), the nose tip, the mouth corners.
const EYES = [
  [33, 133, 159, 145],
  [362, 263, 386, 374],
];
const NOSE = 1;
const MOUTH = [61, 291];

// The tracker's points sit slightly differently from the face finder the swap model was trained
// with; these offsets (in units of the eye distance, measured on the benchmark's tuning faces)
// move them onto it. Order: left eye, right eye, nose, left mouth, right mouth (as seen in the image).
export const KPS_OFFSET = [
  [-0.019, -0.017],
  [0.023, -0.014],
  [-0.001, -0.043],
  [0.01, -0.002],
  [-0.004, -0.011],
];

// points: Float32Array of x,y pixel pairs (at least 468 of them). Returns [[x, y] x 5] in the swap model's
// order (image-left eye first), or null when there aren't enough points.
export function fivePoints(points, offset = KPS_OFFSET) {
  if (!points || points.length < MESH_POINT_COUNT * 2) return null;
  const at = (i) => [points[i * 2], points[i * 2 + 1]];
  const mean = (ids) => {
    let x = 0;
    let y = 0;
    for (const i of ids) {
      x += points[i * 2];
      y += points[i * 2 + 1];
    }
    return [x / ids.length, y / ids.length];
  };
  let eyes = EYES.map(mean);
  let mouth = MOUTH.map(at);
  // Image-left first, measured along the eye line so a tilted head keeps the right order.
  const ax = eyes[1][0] - eyes[0][0];
  const ay = eyes[1][1] - eyes[0][1];
  if (ax < 0) eyes = [eyes[1], eyes[0]];
  const dir = ax < 0 ? [-ax, -ay] : [ax, ay];
  if ((mouth[1][0] - mouth[0][0]) * dir[0] + (mouth[1][1] - mouth[0][1]) * dir[1] < 0) mouth = [mouth[1], mouth[0]];
  const kps = [eyes[0], eyes[1], at(NOSE), mouth[0], mouth[1]];
  // Offsets are in a frame aligned with the eyes: x along the eye line, y down the face.
  const len = Math.hypot(dir[0], dir[1]) || 1;
  const ux = [dir[0] / len, dir[1] / len];
  const uy = [-ux[1], ux[0]];
  return kps.map(([x, y], i) => [
    x + (offset[i][0] * ux[0] + offset[i][1] * uy[0]) * len,
    y + (offset[i][0] * ux[1] + offset[i][1] * uy[1]) * len,
  ]);
}

export const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
export const INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];

// The same alignment the face engine uses (src/main/faceswap/face-math.js), for the studio side.
const ARCFACE_DST = [
  [38.2946, 51.6963],
  [73.5318, 51.5014],
  [56.0252, 71.7366],
  [41.5493, 92.3655],
  [70.7299, 92.2041],
];

export function alignmentTarget(size) {
  const ratio = size % 112 === 0 ? size / 112 : size / 128;
  const diffX = size % 112 === 0 ? 0 : 8 * ratio;
  return ARCFACE_DST.map(([x, y]) => [x * ratio + diffX, y * ratio]);
}

// Least-squares similarity (rotation, uniform scale, shift) taking src points onto dst points.
// Returns [a, c, e, b, d, f]: x' = a*x + c*y + e, y' = b*x + d*y + f.
export function similarity(src, dst) {
  const n = src.length;
  let msx = 0;
  let msy = 0;
  let mdx = 0;
  let mdy = 0;
  for (let i = 0; i < n; i += 1) {
    msx += src[i][0] / n;
    msy += src[i][1] / n;
    mdx += dst[i][0] / n;
    mdy += dst[i][1] / n;
  }
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
  if (!den) return null;
  const a = numA / den;
  const b = numB / den;
  return [a, -b, mdx - (a * msx - b * msy), b, a, mdy - (b * msx + a * msy)];
}

export function invertAffine([a, c, e, b, d, f]) {
  const det = a * d - b * c;
  return [d / det, -c / det, (c * f - d * e) / det, -b / det, a / det, (b * e - a * f) / det];
}

export function composeAffine(m, n) {
  // m after n
  return [
    m[0] * n[0] + m[1] * n[3],
    m[0] * n[1] + m[1] * n[4],
    m[0] * n[2] + m[1] * n[5] + m[2],
    m[3] * n[0] + m[4] * n[3],
    m[3] * n[1] + m[4] * n[4],
    m[3] * n[2] + m[4] * n[5] + m[5],
  ];
}
