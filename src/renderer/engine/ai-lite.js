// Lite engine: "wear" the face from a photo with no generative AI, so it runs on almost any PC.
//
//   1. Track 478 face points with MediaPipe (small, runs on the CPU), smoothed with a one-euro filter.
//   2. Warp the photo's face onto yours with a fixed triangle mesh built once from the photo,
//      leaving holes for your real eyes and mouth so blinks, speech and teeth stay yours.
//   3. In a WebGL shader: match the photo's colours to your camera in LAB space, take the room's
//      shading from your camera (frequency separation), add back your skin highlights, and feather
//      the edges. Triangles that turn away from the camera fade out instead of smearing.
import { FaceLandmarker } from '@mediapipe/tasks-vision';
import Delaunator from 'delaunator';
import { OUTPUT_HEIGHT, OUTPUT_WIDTH } from './state.js';

const FRESH_MS = 1000;
// How much of the camera's lighting and skin detail is laid over the photo's face.
const TUNE = { shading: 0.2, detail: 0.05, blurDiv: 10, contrast: 0.5 };
const MESH_POINTS = 468; // 0..467 are the face; 468..477 are the irises
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const RIGHT_EYE = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7];
const LEFT_EYE = [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249];
const INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const HOLES = [RIGHT_EYE, LEFT_EYE, INNER_LIPS];

// ---------- small geometry helpers ----------

function inPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i, i += 1) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
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

// ---------- one-euro filter: steady when still, instant when you move ----------

class OneEuro {
  constructor({ minCutoff = 1.2, beta = 0.06, dCutoff = 1.0 } = {}) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.x = null;
    this.dx = null;
    this.t = 0;
  }

  reset() {
    this.x = null;
  }

  static alpha(cutoff, dt) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(values, tMs) {
    if (!this.x || this.x.length !== values.length) {
      this.x = Float32Array.from(values);
      this.dx = new Float32Array(values.length);
      this.t = tMs;
      return this.x;
    }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    this.t = tMs;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    for (let i = 0; i < values.length; i += 1) {
      const d = (values[i] - this.x[i]) / dt;
      this.dx[i] = ad * d + (1 - ad) * this.dx[i];
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] = a * values[i] + (1 - a) * this.x[i];
    }
    return this.x;
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

void main() {
  vec2 fuv = vec2(gl_FragCoord.x / uViewport.x, 1.0 - gl_FragCoord.y / uViewport.y);
  vec3 src = transfer(texture(uSrc, vSrcUV).rgb);
  vec3 srcLow = transfer(textureLod(uSrc, vSrcUV, uSrcLod).rgb);
  vec3 frame = texture(uFrame, fuv).rgb;
  vec3 frameLow = textureLod(uFrame, fuv, uFrameLod).rgb;
  // Photo detail + the room's shading on your face + your live skin highlights.
  vec3 col = src + (frameLow - srcLow) * uShading + (frame - frameLow) * uDetail;
  outColor = vec4(clamp(col, 0.0, 1.0), vAlpha * uStrength);
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
    this.frame = 0;
    this.lastTs = 0;
    this.lastResultAt = 0;
    this.faceFound = false;
    this.stats = { ms: 0, fps: 0 };
    this.times = [];
    this.smoother = new OneEuro();

    this.input = document.createElement('canvas');
    this.input.width = OUTPUT_WIDTH;
    this.input.height = OUTPUT_HEIGHT;
    this.inputCtx = this.input.getContext('2d', { alpha: false, willReadFrequently: true });
    this.small = document.createElement('canvas');
    this.small.width = 320;
    this.small.height = 180;
    this.smallCtx = this.small.getContext('2d', { willReadFrequently: true });

    this.canvas = document.createElement('canvas');
    this.canvas.width = OUTPUT_WIDTH;
    this.canvas.height = OUTPUT_HEIGHT;
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
    for (const name of ['uSrc', 'uFrame', 'uViewport', 'uSrcMean', 'uSrcStd', 'uDstMean', 'uDstStd', 'uSrcLod', 'uFrameLod', 'uShading', 'uDetail', 'uStrength']) {
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

    // Fixed topology from the photo: never re-triangulated, so the mesh can't flicker.
    const del = new Delaunator(points);
    const tris = [];
    for (let t = 0; t < del.triangles.length; t += 3) {
      const [a, b, c] = [del.triangles[t], del.triangles[t + 1], del.triangles[t + 2]];
      const cx = (points[a * 2] + points[b * 2] + points[c * 2]) / 3;
      const cy = (points[a * 2 + 1] + points[b * 2 + 1] + points[c * 2 + 1]) / 3;
      if (!inPolygon(cx, cy, oval)) continue; // outside the face
      if (holes.some((h) => inPolygon(cx, cy, h))) continue; // your real eyes and mouth show through
      tris.push(a, b, c);
    }

    // Soft edges: fade towards the face outline and around the eye and mouth holes.
    const faceWidth = Math.max(...oval.map((p) => p[0])) - Math.min(...oval.map((p) => p[0]));
    const feather = new Float32Array(MESH_POINTS);
    for (let i = 0; i < MESH_POINTS; i += 1) {
      const [x, y] = pt(i);
      const edge = inPolygon(x, y, oval) ? distToPolygonEdge(x, y, oval) / faceWidth : 0;
      let a = smoothstep(0.0, 0.09, edge);
      for (const h of holes) a = Math.min(a, smoothstep(0.0, 0.035, distToPolygonEdge(x, y, h) / faceWidth));
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

  // Called by the compositor every frame.
  tick(drawSource) {
    if (this.state !== 'generating' || !this.ready || !this.source) return;
    const started = performance.now();
    drawSource(this.inputCtx, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    const ts = Math.max(this.lastTs + 1, Math.floor(started));
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

    let points = null;
    if (landmarks) {
      const raw = new Float32Array(MESH_POINTS * 2);
      for (let i = 0; i < MESH_POINTS; i += 1) {
        raw[i * 2] = landmarks[i].x * OUTPUT_WIDTH;
        raw[i * 2 + 1] = landmarks[i].y * OUTPUT_HEIGHT;
      }
      points = this.smoother.filter(raw, started);
      if (this.frame % 4 === 0 || !this.dstStats) this.updateDstStats(points);
    } else {
      this.smoother.reset();
    }
    this.faceFound = Boolean(points);
    this.render(points);
    this.frame += 1;

    const now = performance.now();
    this.lastResultAt = now;
    this.stats.ms = Math.round(now - started);
    this.times.push(now);
    while (this.times.length && now - this.times[0] > 1000) this.times.shift();
    this.stats.fps = this.times.length;
  }

  updateDstStats(points) {
    const sx = this.small.width / OUTPUT_WIDTH;
    const sy = this.small.height / OUTPUT_HEIGHT;
    this.smallCtx.drawImage(this.input, 0, 0, this.small.width, this.small.height);
    const data = this.smallCtx.getImageData(0, 0, this.small.width, this.small.height).data;
    const oval = FACE_OVAL.map((i) => [points[i * 2] * sx, points[i * 2 + 1] * sy]);
    const stats = labStats(data, this.small.width, this.small.height, oval);
    if (stats) this.dstStats = stats;
  }

  render(points) {
    const gl = this.gl;
    gl.viewport(0, 0, OUTPUT_WIDTH, OUTPUT_HEIGHT);
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
    if (!points || !this.dstStats) return;

    // Then the photo's face, warped onto yours.
    const src = this.source;
    const tris = src.tris;
    const triCount = tris.length / 3;
    const ovalTarget = FACE_OVAL.map((i) => points[i * 2]);
    const targetWidth = Math.max(...ovalTarget) - Math.min(...ovalTarget);
    const scale2 = (targetWidth / src.faceWidth) ** 2;
    const verts = new Float32Array(triCount * 3 * 5);
    let o = 0;
    for (let t = 0; t < triCount; t += 1) {
      const a = tris[t * 3];
      const b = tris[t * 3 + 1];
      const c = tris[t * 3 + 2];
      // Triangles that shrink or flip are turning away from the camera: fade them out.
      const ratio = triArea(points, a, b, c) / (src.srcAreas[t] * scale2 || 1);
      const facing = smoothstep(0.15, 0.55, ratio);
      for (const i of [a, b, c]) {
        verts[o++] = points[i * 2];
        verts[o++] = points[i * 2 + 1];
        verts[o++] = src.points[i * 2] / src.width;
        verts[o++] = src.points[i * 2 + 1] / src.height;
        verts[o++] = src.feather[i] * facing;
      }
    }

    gl.useProgram(this.faceProgram);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.srcTex);
    gl.uniform1i(this.loc.uSrc, 0);
    gl.uniform1i(this.loc.uFrame, 1);
    gl.uniform2f(this.loc.uViewport, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    gl.uniform3fv(this.loc.uSrcMean, src.stats.mean);
    gl.uniform3fv(this.loc.uSrcStd, src.stats.std);
    gl.uniform3fv(this.loc.uDstMean, this.dstStats.mean);
    gl.uniform3fv(this.loc.uDstStd, this.dstStats.std.map((d, c) => d + (src.stats.std[c] - d) * TUNE.contrast));
    // Blur radius about 1/BLUR_DIV of the face, in each image's own scale: only broad lighting moves over.
    gl.uniform1f(this.loc.uSrcLod, Math.max(0, Math.log2(src.faceWidth / TUNE.blurDiv)));
    gl.uniform1f(this.loc.uFrameLod, Math.max(0, Math.log2(targetWidth / TUNE.blurDiv)));
    gl.uniform1f(this.loc.uShading, TUNE.shading);
    gl.uniform1f(this.loc.uDetail, TUNE.detail);
    gl.uniform1f(this.loc.uStrength, 1.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.faceBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.DYNAMIC_DRAW);
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
