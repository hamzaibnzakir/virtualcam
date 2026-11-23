// Hair and outfit from the reference photo, drawn over the camera picture (no generative AI):
//   - hair: the photo's hair is cut out once (MediaPipe multiclass segmentation) and placed on your
//     head every frame, following the face tracker;
//   - outfit: the photo's clothes are cut out once; your own clothes are found in the camera picture
//     a few times a second, and the photo's clothes are stretched over them, kept inside their outline;
//   - your own hair (optional, with the photo's hair): found in the camera picture with the same tool
//     and painted over with the background behind you (remembered from frames where it showed),
//     except over your face.
// Both engines that track your face (Lite and Local AI) use this.
import { FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';
import { MESH_POINT_COUNT, FACE_OVAL, fivePoints, similarity } from './face-points.js';

const BACKGROUND = 0;
const HAIR = 1;
const CLOTHES = 4;
const SEG_SIZE = 256;
const HAIR_GRID = 320; // same as SEG_INPUT: the hair fill reuses the segmenter's copy of the picture
const SEG_INPUT = 320; // width of the camera copy the segmenter gets // width of the grid your own hair is painted over on
const PHOTO_MAX = 1024;
const CAMERA_SEG_EVERY_MS = 150; // how often your clothes (and hair) are found again
const BOX_EASE = 0.35; // smoothing of the outfit's placement between finds

function smoothstep(lo, hi, v) {
  const t = Math.min(1, Math.max(0, (v - lo) / (hi - lo)));
  return t * t * (3 - 2 * t);
}

// A canvas of `src` whose alpha is the segmentation confidence for `cls` (mask covers `box` of src).
function cutOut(src, conf, size, box, lo, hi) {
  const mask = document.createElement('canvas');
  mask.width = size;
  mask.height = size;
  const mctx = mask.getContext('2d');
  const img = mctx.createImageData(size, size);
  let any = 0;
  for (let i = 0; i < conf.length; i += 1) {
    const a = smoothstep(lo, hi, conf[i]);
    img.data[i * 4 + 3] = Math.round(a * 255);
    any += a;
  }
  mctx.putImageData(img, 0, 0);
  const out = document.createElement('canvas');
  out.width = src.width;
  out.height = src.height;
  const ctx = out.getContext('2d');
  ctx.drawImage(src, 0, 0);
  ctx.globalCompositeOperation = 'destination-in';
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(mask, box.x, box.y, box.side, box.side);
  return { canvas: out, area: any / (size * size) };
}

// Bounding box (in mask cells) of where confidence passes `level`, or null.
function maskBox(conf, size, level = 0.5) {
  let minX = size;
  let minY = size;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      if (conf[y * size + x] < level) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return maxX < 0 ? null : { x0: minX, y0: minY, x1: maxX + 1, y1: maxY + 1 };
}

// Fills cells with known[i] = 0 from the known ones around them (push-pull over a pyramid).
function fillHoles(col, known, w, h) {
  if (w <= 1 || h <= 1) return;
  const w2 = Math.ceil(w / 2);
  const h2 = Math.ceil(h / 2);
  const c2 = new Float32Array(w2 * h2 * 3);
  const k2 = new Float32Array(w2 * h2);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      if (!known[i]) continue;
      const j = (y >> 1) * w2 + (x >> 1);
      const wt = known[i];
      k2[j] += wt;
      for (let c = 0; c < 3; c += 1) c2[j * 3 + c] += col[i * 3 + c] * wt;
    }
  }
  for (let j = 0; j < w2 * h2; j += 1) {
    if (k2[j] > 0) for (let c = 0; c < 3; c += 1) c2[j * 3 + c] /= k2[j];
    k2[j] = Math.min(1, k2[j]);
  }
  let anyMissing = false;
  for (let j = 0; j < k2.length; j += 1) if (k2[j] < 1) anyMissing = true;
  if (anyMissing) fillHoles(c2, k2, w2, h2);
  // Pull back up with bilinear sampling of the coarser level (smooth, no blocks).
  for (let y = 0; y < h; y += 1) {
    const fy = Math.min(h2 - 1, Math.max(0, (y + 0.5) / 2 - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(h2 - 1, y0 + 1);
    const ty = fy - y0;
    for (let x = 0; x < w; x += 1) {
      const i = y * w + x;
      const kn = Math.min(1, known[i]);
      if (kn >= 1) continue;
      const fx = Math.min(w2 - 1, Math.max(0, (x + 0.5) / 2 - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(w2 - 1, x0 + 1);
      const tx = fx - x0;
      for (let c = 0; c < 3; c += 1) {
        const v =
          (c2[(y0 * w2 + x0) * 3 + c] * (1 - tx) + c2[(y0 * w2 + x1) * 3 + c] * tx) * (1 - ty) +
          (c2[(y1 * w2 + x0) * 3 + c] * (1 - tx) + c2[(y1 * w2 + x1) * 3 + c] * tx) * ty;
        col[i * 3 + c] = col[i * 3 + c] * kn + v * (1 - kn);
      }
    }
  }
}

// Height of the eyes (mesh points 33 and 263), in picture pixels.
function eyeLine(points) {
  return points ? (points[33 * 2 + 1] + points[263 * 2 + 1]) / 2 : null;
}

// Where your hair can be: a box around your face, wide enough for big hair and long enough for hair
// down to your chest (in picture pixels).
function headArea(points) {
  if (!points) return null;
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < points.length; i += 2) {
    x0 = Math.min(x0, points[i]);
    x1 = Math.max(x1, points[i]);
    y0 = Math.min(y0, points[i + 1]);
    y1 = Math.max(y1, points[i + 1]);
  }
  const w = x1 - x0;
  const h = y1 - y0;
  return {
    x0: x0 - w * 1.2,
    x1: x1 + w * 1.2,
    y0: y0 - h * 1.2,
    y1: y1 + h * 2.2,
    // Hair starts at the top and sides of your face.
    seed: { x0: x0 - w * 0.3, x1: x1 + w * 0.3, y0: y0 - h * 0.6, y1: y0 + h * 0.6 },
  };
}

// Grows a 0/1 mask by r cells.
function dilate(mask, w, h, r) {
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let on = 0;
      for (let d = -r; d <= r && !on; d += 1) {
        const xx = x + d;
        if (xx >= 0 && xx < w && mask[y * w + xx]) on = 1;
      }
      tmp[y * w + x] = on;
    }
  }
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      let on = 0;
      for (let d = -r; d <= r && !on; d += 1) {
        const yy = y + d;
        if (yy >= 0 && yy < h && tmp[yy * w + x]) on = 1;
      }
      out[y * w + x] = on;
    }
  }
  return out;
}

export class LookLayers {
  constructor(api) {
    this.api = api;
    this.photo = null; // the prepared reference photo
    this.photoKey = null;
    this.camMask = document.createElement('canvas');
    this.camMask.width = SEG_SIZE;
    this.camMask.height = SEG_SIZE;
    this.camMaskCtx = this.camMask.getContext('2d');
    this.camBox = null; // your clothes, in output pixels (smoothed)
    this.lastSegAt = -Infinity;
    this.layer = document.createElement('canvas');
    this.layerCtx = this.layer.getContext('2d');
    this.segInput = document.createElement('canvas'); // small copy of the camera picture to segment
    // Kept on the processor: its pixels are read for the hair fill (reading a graphics-chip canvas stalls).
    this.segInputCtx = this.segInput.getContext('2d', { willReadFrequently: true });
    this.small = document.createElement('canvas'); // the camera picture at mask size
    this.smallCtx = this.small.getContext('2d', { willReadFrequently: true });
    this.hairFill = document.createElement('canvas'); // background painted where your hair is
    this.hairFillCtx = this.hairFill.getContext('2d');
    this.hairSoft = document.createElement('canvas'); // the same, softened once per update (not per frame)
    this.hairSoftCtx = this.hairSoft.getContext('2d');
    this.hairFillOn = false;
    this.plate = null; // remembered background colours (r, g, b per mask cell)
    this.seen = null;
    this.error = '';
  }

  async fileset() {
    if (!this.files) {
      const wasm = await this.api.readAsset('mediapipe/vision_wasm_internal.wasm');
      this.files = {
        wasmLoaderPath: new URL('mediapipe/vision_wasm_internal.js', document.baseURI).href,
        wasmBinaryPath: URL.createObjectURL(new Blob([wasm], { type: 'application/wasm' })),
      };
    }
    return this.files;
  }

  async segModel() {
    if (!this.segBytes) this.segBytes = new Uint8Array(await this.api.readAsset('mediapipe/selfie_multiclass_256x256.tflite'));
    return this.segBytes;
  }

  async ensurePhotoTools() {
    if (this.imageLm && this.imageSeg) return;
    const files = await this.fileset();
    const lmModel = new Uint8Array(await this.api.readAsset('mediapipe/face_landmarker.task'));
    this.imageLm = await FaceLandmarker.createFromOptions(files, {
      baseOptions: { modelAssetBuffer: lmModel, delegate: 'CPU' },
      runningMode: 'IMAGE',
      numFaces: 1,
    });
    this.imageSeg = await ImageSegmenter.createFromOptions(files, {
      baseOptions: { modelAssetBuffer: await this.segModel(), delegate: 'CPU' },
      runningMode: 'IMAGE',
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
  }

  async ensureCameraTool() {
    if (this.videoSeg) return;
    const files = await this.fileset();
    const make = (delegate) =>
      ImageSegmenter.createFromOptions(files, {
        baseOptions: { modelAssetBuffer: this.segBytes, delegate },
        runningMode: 'VIDEO',
        outputConfidenceMasks: true,
        outputCategoryMask: false,
      });
    await this.segModel();
    // Starts on the processor: it runs a few times a second on a small picture, and its masks are
    // then already in memory. If the processor turns out slow at it, it moves to the graphics chip
    // (see pickSegDevice).
    this.makeSeg = make;
    this.segRunTimes = [];
    try {
      this.videoSeg = await make('CPU');
      this.segDelegate = 'CPU';
    } catch {
      this.videoSeg = await make('GPU');
      this.segDelegate = 'GPU';
    }
  }

  // After a few runs (the first one warms up), keep whichever device finds you faster.
  async pickSegDevice(ms) {
    if (this.segSwitching || this.segDecided) return;
    this.segRunTimes.push(ms);
    if (this.segRunTimes.length < 4) return;
    const avg = this.segRunTimes.slice(1).reduce((a, b) => a + b, 0) / (this.segRunTimes.length - 1);
    this.segAvg = { ...(this.segAvg || {}), [this.segDelegate]: Math.round(avg) };
    const other = this.segDelegate === 'CPU' ? 'GPU' : 'CPU';
    if (this.segDelegate === 'CPU' && avg > 40 && !this.segAvg.GPU) {
      this.segSwitching = true;
      try {
        const seg = await this.makeSeg(other);
        if (this.videoSeg) this.videoSeg.close();
        this.videoSeg = seg;
        this.segDelegate = other;
        this.segRunTimes = [];
      } catch {
        this.segDecided = true;
      } finally {
        this.segSwitching = false;
      }
      return;
    }
    if (this.segAvg.CPU && this.segAvg.GPU && this.segDelegate === 'GPU' && this.segAvg.GPU > this.segAvg.CPU) {
      // The processor was faster after all: go back.
      try {
        const seg = await this.makeSeg('CPU');
        this.videoSeg.close();
        this.videoSeg = seg;
        this.segDelegate = 'CPU';
      } catch {
        // stay
      }
    }
    this.segDecided = true;
  }

  // Reads the photo once: its face position, hair and clothes. Returns false when it has no face.
  async preparePhoto(key, blob, { outfit = false } = {}) {
    if (this.photoKey === key && this.photo) {
      if (outfit) await this.ensureCameraTool();
      return true;
    }
    this.photo = null;
    this.photoKey = key;
    if (!blob) return false;
    await this.ensurePhotoTools();
    if (outfit) await this.ensureCameraTool();
    const bitmap = await createImageBitmap(blob);
    const scale = Math.min(1, PHOTO_MAX / Math.max(bitmap.width, bitmap.height));
    const src = document.createElement('canvas');
    src.width = Math.round(bitmap.width * scale);
    src.height = Math.round(bitmap.height * scale);
    src.getContext('2d').drawImage(bitmap, 0, 0, src.width, src.height);
    bitmap.close();

    const lm = this.imageLm.detect(src);
    const face = lm.faceLandmarks && lm.faceLandmarks[0];
    if (!face) return false;
    const pts = new Float32Array(MESH_POINT_COUNT * 2);
    for (let i = 0; i < MESH_POINT_COUNT; i += 1) {
      pts[i * 2] = face[i].x * src.width;
      pts[i * 2 + 1] = face[i].y * src.height;
    }
    const kps = fivePoints(pts);

    // Face size and position in the photo, to frame the two cut-outs.
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < MESH_POINT_COUNT; i += 1) {
      minX = Math.min(minX, pts[i * 2]);
      maxX = Math.max(maxX, pts[i * 2]);
      minY = Math.min(minY, pts[i * 2 + 1]);
      maxY = Math.max(maxY, pts[i * 2 + 1]);
    }
    const fw = maxX - minX;
    const cx = (minX + maxX) / 2;
    // Hair: a square around the head. Clothes: a bigger square from the chin down.
    const hairSide = fw * 3;
    const hairBox = { x: cx - hairSide / 2, y: minY - fw * 1.1, side: hairSide };
    const bodySide = Math.min(Math.max(src.width, src.height) * 1.2, fw * 4.5);
    const bodyBox = { x: cx - bodySide / 2, y: maxY - fw * 0.35, side: bodySide };
    const hairSeg = this.segmentSquare(src, hairBox);
    const bodySeg = this.segmentSquare(src, bodyBox);
    if (!hairSeg || !bodySeg) return false;
    const hair = cutOut(src, hairSeg.masks[HAIR], hairSeg.size, hairBox, 0.35, 0.75);
    // Clothes, plus 'other' (patterned fabric, accessories) below the chin.
    const size = bodySeg.size;
    const chinRow = Math.max(0, Math.round(((maxY - bodyBox.y) / bodySide) * size));
    const clothesConf = new Float32Array(size * size);
    const other = bodySeg.masks[CLOTHES + 1];
    for (let i = 0; i < clothesConf.length; i += 1) {
      const row = Math.floor(i / size);
      clothesConf[i] = Math.max(bodySeg.masks[CLOTHES][i], other && row > chinRow ? other[i] : 0);
    }
    const clothes = cutOut(src, clothesConf, size, bodyBox, 0.4, 0.7);
    const cb = maskBox(clothesConf, size);
    const cell = bodySide / size;
    const clothesBox = cb
      ? { x0: bodyBox.x + cb.x0 * cell, y0: bodyBox.y + cb.y0 * cell, x1: bodyBox.x + cb.x1 * cell, y1: bodyBox.y + cb.y1 * cell }
      : null;
    this.measured = { hairArea: Number(hair.area.toFixed(4)), clothesArea: Number(clothes.area.toFixed(4)), clothesBox: Boolean(clothesBox) };
    this.photo = {
      kps,
      hair: hair.area > 0.002 ? hair.canvas : null,
      clothes: clothes.area > 0.003 && clothesBox ? clothes.canvas : null,
      clothesBox,
    };
    return true;
  }

  // Segments a square of the photo (parts outside it count as black); returns each class's confidence.
  segmentSquare(src, box) {
    const square = document.createElement('canvas');
    square.width = SEG_SIZE;
    square.height = SEG_SIZE;
    const sctx = square.getContext('2d');
    sctx.fillStyle = '#000';
    sctx.fillRect(0, 0, SEG_SIZE, SEG_SIZE);
    sctx.drawImage(src, box.x, box.y, box.side, box.side, 0, 0, SEG_SIZE, SEG_SIZE);
    const res = this.imageSeg.segment(square);
    const masks = res.confidenceMasks;
    if (!masks || masks.length <= CLOTHES) return null;
    const out = { size: masks[0].width, masks: masks.map((m) => Float32Array.from(m.getAsFloat32Array())) };
    masks.forEach((m) => m.close && m.close());
    return out;
  }

  get hasHair() {
    return Boolean(this.photo && this.photo.hair);
  }

  get hasOutfit() {
    return Boolean(this.photo && this.photo.clothes);
  }

  // Finds your clothes (and your hair, when it should be hidden) in the camera picture, a few times a
  // second.
  findCameraClothes(frame, width, height, t, chinY = null, { clothes = true, hair = false, eyeY = null, head = null } = {}) {
    // Every CAMERA_SEG_EVERY_MS, or less often when finding takes long (no graphics chip), so it
    // never takes more than about a fifth of the time.
    if (!this.videoSeg || t - this.lastSegAt < Math.max(CAMERA_SEG_EVERY_MS, (this.segMs || 0) * 5)) return;
    this.lastSegAt = t;
    const segStart = performance.now();
    try {
      this.segmentCamera(frame, width, height, chinY, { clothes, hair, eyeY, head });
    } finally {
      // All of it (finding, reading the result back, the maths), so the pacing above is honest.
      this.segMs = performance.now() - segStart;
      this.pickSegDevice(this.segMs);
    }
  }

  segmentCamera(frame, width, height, chinY, { clothes, hair, eyeY, head }) {
    // The model looks at 256 px anyway: give it a small copy, so its masks are small too (reading
    // full-size masks back from the graphics chip was the slow part).
    const sw = Math.min(width, SEG_INPUT);
    const sh = Math.max(1, Math.round((height * sw) / width));
    if (this.segInput.width !== sw || this.segInput.height !== sh) {
      this.segInput.width = sw;
      this.segInput.height = sh;
    }
    this.segInputCtx.drawImage(frame, 0, 0, sw, sh);
    let res;
    this.camRuns = (this.camRuns || 0) + 1;
    const s0 = performance.now();
    try {
      res = this.videoSeg.segmentForVideo(this.segInput, Math.floor(performance.now()));
      this.segRunMs = performance.now() - s0;
    } catch (err) {
      this.camError = String((err && err.message) || err).slice(0, 200);
      return;
    }
    const masks = res.confidenceMasks;
    if (!masks || masks.length <= CLOTHES) return;
    const mw = masks[CLOTHES].width;
    const mh = masks[CLOTHES].height;
    if (hair) {
      // Hair is only looked for around your head (stray dark patches elsewhere aren't hair).
      const norm = (b) => b && { x0: b.x0 / width, y0: b.y0 / height, x1: b.x1 / width, y1: b.y1 / height };
      const area = head && { ...norm(head), seed: norm(head.seed) };
      this.updateHairFill(frame, masks, mw, mh, eyeY === null ? 1 : eyeY / height, area);
    }
    if (!clothes) {
      masks.forEach((m) => m.close && m.close());
      return;
    }
    const clothesConf = masks[CLOTHES].getAsFloat32Array();
    const otherConf = masks.length > CLOTHES + 1 ? masks[CLOTHES + 1].getAsFloat32Array() : null;
    // Clothes, plus 'other' (patterned or plain fabric the model isn't sure of) below your chin.
    const chinRow = chinY === null ? mh : Math.round((chinY / height) * mh);
    const conf = new Float32Array(mw * mh);
    for (let i = 0; i < conf.length; i += 1) {
      conf[i] = otherConf && i >= chinRow * mw ? Math.max(clothesConf[i], otherConf[i]) : clothesConf[i];
    }
    if (this.camMask.width !== mw || this.camMask.height !== mh) {
      this.camMask.width = mw;
      this.camMask.height = mh;
    }
    const img = this.camMaskCtx.createImageData(mw, mh);
    let minX = mw;
    let minY = mh;
    let maxX = -1;
    let maxY = -1;
    let on = 0;
    for (let y = 0; y < mh; y += 1) {
      for (let x = 0; x < mw; x += 1) {
        const c = conf[y * mw + x];
        img.data[(y * mw + x) * 4 + 3] = Math.round(smoothstep(0.35, 0.7, c) * 255);
        if (c < 0.5) continue;
        on += 1;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
    masks.forEach((m) => m.close && m.close());
    this.camMaskCtx.putImageData(img, 0, 0);
    this.camOn = Number((on / (mw * mh)).toFixed(4));
    this.camSize = [mw, mh];
    this.camFound = maxX < 0 ? 0 : Number((((maxX - minX) * (maxY - minY)) / (mw * mh)).toFixed(4));
    if (maxX < 0 || (maxX - minX) * (maxY - minY) < mw * mh * 0.003) {
      this.camBox = null;
      return;
    }
    const sx = width / mw;
    const sy = height / mh;
    const found = { x0: minX * sx, y0: minY * sy, x1: (maxX + 1) * sx, y1: (maxY + 1) * sy };
    if (!this.camBox) this.camBox = found;
    else for (const k of ['x0', 'y0', 'x1', 'y1']) this.camBox[k] += (found[k] - this.camBox[k]) * BOX_EASE;
  }

  // Paints the background over your hair: remembered background where it was seen, else the nearby
  // background colours spread inwards. Runs with each camera segmentation.
  // eyeLine: 0..1 down the picture; your face's skin helps fill only above it (the hairline).
  updateHairFill(frame, masks, maskW, maskH, eyeLine = 1, area = null) {
    const fillStart = performance.now();
    // Work on a small grid (the masks can be full camera size).
    const mw = Math.min(maskW, HAIR_GRID);
    const mh = Math.max(1, Math.round((maskH * mw) / maskW));
    if (this.small.width !== mw || this.small.height !== mh) {
      this.small.width = mw;
      this.small.height = mh;
      this.hairFill.width = mw;
      this.hairFill.height = mh;
      this.plate = new Float32Array(mw * mh * 3);
      this.seen = new Uint8Array(mw * mh);
    }
    const p0 = performance.now();
    let px;
    if (this.segInput.width === mw && this.segInput.height === mh) {
      px = this.segInputCtx.getImageData(0, 0, mw, mh).data;
    } else {
      this.smallCtx.drawImage(this.segInput, 0, 0, mw, mh);
      px = this.smallCtx.getImageData(0, 0, mw, mh).data;
    }
    const p1 = performance.now();
    const sample = (m) => {
      const full = m.getAsFloat32Array();
      const out = new Float32Array(mw * mh);
      for (let y = 0; y < mh; y += 1) {
        const row = Math.min(maskH - 1, Math.floor(((y + 0.5) * maskH) / mh)) * maskW;
        for (let x = 0; x < mw; x += 1) out[y * mw + x] = full[row + Math.min(maskW - 1, Math.floor(((x + 0.5) * maskW) / mw))];
      }
      return out;
    };
    const bg = sample(masks[BACKGROUND]);
    const hairConf = sample(masks[HAIR]);
    const body = sample(masks[HAIR + 1]); // body skin
    const faceSkin = sample(masks[HAIR + 2]);
    const clothesConf = sample(masks[CLOTHES]);
    const p2 = performance.now();
    const n = mw * mh;
    const skinRows = Math.round(eyeLine * mh) * mw;
    // Remember the background wherever it's clearly visible.
    const known = new Float32Array(n);
    const col = new Float32Array(n * 3);
    for (let i = 0; i < n; i += 1) {
      if (bg[i] > 0.85) {
        const k = this.seen[i] ? 0.25 : 1;
        for (let c = 0; c < 3; c += 1) this.plate[i * 3 + c] += (px[i * 4 + c] - this.plate[i * 3 + c]) * k;
        this.seen[i] = 1;
      }
      if (this.seen[i]) {
        known[i] = 1;
        for (let c = 0; c < 3; c += 1) col[i * 3 + c] = this.plate[i * 3 + c];
      } else if (hairConf[i] < 0.2 && (body[i] > 0.6 || clothesConf[i] > 0.6 || (faceSkin[i] > 0.6 && i < skinRows))) {
        // Your skin and clothes as they are now (hair over your shoulders becomes shoulders, and the
        // edge of your forehead becomes forehead).
        known[i] = 1;
        for (let c = 0; c < 3; c += 1) col[i * 3 + c] = px[i * 4 + c];
      }
    }
    fillHoles(col, known, mw, mh);
    // Where to paint: your hair, grown by a few cells so no strands are left at its edge.
    const hairMask = new Uint8Array(n);
    let any = 0;
    const box = (b) => b && [Math.floor(b.x0 * mw), Math.ceil(b.x1 * mw), Math.floor(b.y0 * mh), Math.ceil(b.y1 * mh)];
    const [ax0, ax1, ay0, ay1] = box(area) || [0, mw, 0, mh];
    const inside = (x, y, b) => x >= b[0] && x < b[1] && y >= b[2] && y < b[3];
    const isHair = (i) => {
      const x = i % mw;
      const y = (i - x) / mw;
      return hairConf[i] > 0.3 && x >= ax0 && x < ax1 && y >= ay0 && y < ay1;
    };
    const seed = area && box(area.seed);
    if (seed) {
      // Only hair joined to the hair around your face counts (dark patches elsewhere aren't yours).
      const queue = [];
      for (let i = 0; i < n; i += 1) {
        const x = i % mw;
        if (inside(x, (i - x) / mw, seed) && isHair(i)) {
          hairMask[i] = 1;
          queue.push(i);
        }
      }
      while (queue.length) {
        const i = queue.pop();
        any += 1;
        const x = i % mw;
        for (const j of [x > 0 ? i - 1 : -1, x < mw - 1 ? i + 1 : -1, i - mw, i + mw]) {
          if (j >= 0 && j < n && !hairMask[j] && isHair(j)) {
            hairMask[j] = 1;
            queue.push(j);
          }
        }
      }
    } else {
      for (let i = 0; i < n; i += 1) {
        if (isHair(i)) {
          hairMask[i] = 1;
          any += 1;
        }
      }
    }
    const grown = dilate(hairMask, mw, mh, Math.max(1, Math.round(mw / 100)));
    const img = this.hairFillCtx.createImageData(mw, mh);
    for (let i = 0; i < n; i += 1) {
      img.data[i * 4] = col[i * 3];
      img.data[i * 4 + 1] = col[i * 3 + 1];
      img.data[i * 4 + 2] = col[i * 3 + 2];
      img.data[i * 4 + 3] = grown[i] ? 255 : 0;
    }
    this.hairFillCtx.putImageData(img, 0, 0);
    if (this.hairSoft.width !== mw || this.hairSoft.height !== mh) {
      this.hairSoft.width = mw;
      this.hairSoft.height = mh;
    }
    this.hairSoftCtx.clearRect(0, 0, mw, mh);
    this.hairSoftCtx.filter = 'blur(1.5px)';
    this.hairSoftCtx.drawImage(this.hairFill, 0, 0);
    this.hairSoftCtx.filter = 'none';
    this.hairFillOn = any > n * 0.002;
    this.hairCells = any;
    this.fillMs = performance.now() - fillStart;
    this.fillParts = { pixels: Math.round(p1 - p0), masks: Math.round(p2 - p1), maths: Math.round(performance.now() - p2) };
  }

  // Hides your own hair (call before the photo's hair is drawn). Your face is never painted over.
  // segmented: drawOutfit already ran the camera segmentation this frame (with hideHair).
  hideOwnHair(ctx, frame, width, height, t, points, { segmented = false } = {}) {
    if (!this.videoSeg) return;
    if (!segmented) this.findCameraClothes(frame, width, height, t, null, { clothes: false, hair: true, eyeY: eyeLine(points), head: headArea(points) });
    if (!this.hairFillOn) return;
    if (this.layer.width !== width || this.layer.height !== height) {
      this.layer.width = width;
      this.layer.height = height;
    }
    const l = this.layerCtx;
    l.setTransform(1, 0, 0, 1, 0, 0);
    l.globalCompositeOperation = 'source-over';
    l.clearRect(0, 0, width, height);
    l.imageSmoothingEnabled = true;
    l.imageSmoothingQuality = 'high';
    l.drawImage(this.hairSoft, 0, 0, width, height);
    if (points) {
      // Keep the face (from the hairline down) as it is.
      l.globalCompositeOperation = 'destination-out';
      l.beginPath();
      FACE_OVAL.forEach((idx, k) => (k ? l.lineTo(points[idx * 2], points[idx * 2 + 1]) : l.moveTo(points[idx * 2], points[idx * 2 + 1])));
      l.closePath();
      l.fill();
      l.globalCompositeOperation = 'source-over';
    }
    ctx.drawImage(this.layer, 0, 0);
  }

  drawOutfit(ctx, frame, width, height, t, points = null, { hideHair = false } = {}) {
    if (!this.hasOutfit) return;
    let chinY = null;
    if (points) {
      chinY = 0;
      for (let i = 1; i < points.length; i += 2) chinY = Math.max(chinY, points[i]);
    }
    this.findCameraClothes(frame, width, height, t, chinY, { clothes: true, hair: hideHair, eyeY: points ? eyeLine(points) : null, head: headArea(points) });
    const cam = this.camBox;
    const pb = this.photo.clothesBox;
    if (!cam) return;
    if (this.layer.width !== width || this.layer.height !== height) {
      this.layer.width = width;
      this.layer.height = height;
    }
    const l = this.layerCtx;
    l.globalCompositeOperation = 'source-over';
    l.setTransform(1, 0, 0, 1, 0, 0);
    l.clearRect(0, 0, width, height);
    // The photo's clothes, stretched from their box onto your clothes' box...
    const kx = (cam.x1 - cam.x0) / (pb.x1 - pb.x0);
    const ky = (cam.y1 - cam.y0) / (pb.y1 - pb.y0);
    l.setTransform(kx, 0, 0, ky, cam.x0 - pb.x0 * kx, cam.y0 - pb.y0 * ky);
    l.drawImage(this.photo.clothes, 0, 0);
    // ...kept inside your clothes' outline.
    l.setTransform(1, 0, 0, 1, 0, 0);
    l.globalCompositeOperation = 'destination-in';
    l.imageSmoothingQuality = 'high';
    l.drawImage(this.camMask, 0, 0, width, height);
    l.globalCompositeOperation = 'source-over';
    ctx.drawImage(this.layer, 0, 0);
  }

  drawHair(ctx, points, opacity = 1) {
    if (!this.hasHair || !points) return;
    const kps = fivePoints(points);
    const m = kps && similarity(this.photo.kps, kps);
    if (!m) return;
    ctx.save();
    ctx.globalAlpha = opacity;
    ctx.setTransform(m[0], m[3], m[1], m[4], m[2], m[5]);
    ctx.drawImage(this.photo.hair, 0, 0);
    ctx.restore();
  }
}
