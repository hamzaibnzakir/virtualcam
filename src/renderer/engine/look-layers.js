// Hair and outfit from the reference photo, drawn over the camera picture (no generative AI):
//   - hair: the photo's hair is cut out once (MediaPipe multiclass segmentation) and placed on your
//     head every frame, following the face tracker;
//   - outfit: the photo's clothes are cut out once; your own clothes are found in the camera picture
//     a few times a second, and the photo's clothes are stretched over them, kept inside their outline.
// Both engines that track your face (Lite and Local AI) use this.
import { FaceLandmarker, ImageSegmenter } from '@mediapipe/tasks-vision';
import { MESH_POINT_COUNT, fivePoints, similarity } from './face-points.js';

const HAIR = 1;
const CLOTHES = 4;
const SEG_SIZE = 256;
const PHOTO_MAX = 1024;
const CAMERA_SEG_EVERY_MS = 100; // how often your clothes are found again
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
    try {
      this.videoSeg = await make('GPU');
    } catch {
      this.videoSeg = await make('CPU');
    }
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

  // Finds your clothes in the camera picture (a few times a second).
  findCameraClothes(frame, width, height, t, chinY = null) {
    // Every CAMERA_SEG_EVERY_MS, or less often when finding takes long (no graphics chip), so it
    // never takes more than about a fifth of the time.
    if (!this.videoSeg || t - this.lastSegAt < Math.max(CAMERA_SEG_EVERY_MS, (this.segMs || 0) * 5)) return;
    this.lastSegAt = t;
    const segStart = performance.now();
    let res;
    this.camRuns = (this.camRuns || 0) + 1;
    try {
      res = this.videoSeg.segmentForVideo(frame, Math.floor(t));
    } catch (err) {
      this.camError = String((err && err.message) || err).slice(0, 200);
      return;
    }
    const masks = res.confidenceMasks;
    if (!masks || masks.length <= CLOTHES) return;
    this.segMs = performance.now() - segStart;
    const mw = masks[CLOTHES].width;
    const mh = masks[CLOTHES].height;
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

  drawOutfit(ctx, frame, width, height, t, points = null) {
    if (!this.hasOutfit) return;
    let chinY = null;
    if (points) {
      chinY = 0;
      for (let i = 1; i < points.length; i += 2) chinY = Math.max(chinY, points[i]);
    }
    this.findCameraClothes(frame, width, height, t, chinY);
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
