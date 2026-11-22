"""Face finding, alignment and identity, matching what the VirtualCam app does
(src/main/faceswap/detector.js, face-math.js), so training data looks like what the app feeds the model."""
import cv2
import numpy as np
import onnxruntime as ort

ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


def providers():
    have = ort.get_available_providers()
    return [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in have]


def alignment_target(size):
    if size % 112 == 0:
        return ARCFACE_DST * (size / 112)
    ratio = size / 128
    return ARCFACE_DST * ratio + np.array([8 * ratio, 0], dtype=np.float32)


def similarity(src, dst):
    """Least-squares similarity transform src -> dst as a 2x3 matrix (same maths as the app)."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    den = (s ** 2).sum()
    a = (s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]).sum() / den
    b = (s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]).sum() / den
    return np.array([[a, -b, md[0] - (a * ms[0] - b * ms[1])], [b, a, md[1] - (b * ms[0] + a * ms[1])]], np.float32)


def align(rgb, kps, size):
    m = similarity(kps, alignment_target(size))
    return cv2.warpAffine(rgb, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)


class Detector:
    """insightface SCRFD det_10g."""

    STRIDES = (8, 16, 32)

    def __init__(self, path, size=640, threshold=0.5, nms=0.4):
        self.sess = ort.InferenceSession(path, providers=providers())
        self.input = self.sess.get_inputs()[0].name
        self.size, self.threshold, self.nms = size, threshold, nms

    def detect(self, rgb):
        h, w = rgb.shape[:2]
        scale = self.size / max(h, w)
        img = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))))
        canvas = np.zeros((self.size, self.size, 3), np.float32)
        canvas[: img.shape[0], : img.shape[1]] = img
        blob = ((canvas - 127.5) / 128.0).transpose(2, 0, 1)[None].astype(np.float32)
        out = self.sess.run(None, {self.input: blob})
        faces = []
        for level, stride in enumerate(self.STRIDES):
            scores, boxes, kps = out[level].reshape(-1), out[level + 3].reshape(-1, 4), out[level + 6].reshape(-1, 10)
            grid = self.size // stride
            for i in np.where(scores >= self.threshold)[0]:
                cell = i // 2
                cx, cy = (cell % grid) * stride, (cell // grid) * stride
                b = boxes[i] * stride
                box = np.array([cx - b[0], cy - b[1], cx + b[2], cy + b[3]]) / scale
                pts = (np.array([cx, cy]) + kps[i].reshape(5, 2) * stride) / scale
                faces.append((float(scores[i]), box, pts))
        faces.sort(key=lambda f: -f[0])
        kept = []
        for f in faces:
            if all(iou(f[1], k[1]) <= self.nms for k in kept):
                kept.append(f)
        kept.sort(key=lambda f: -(f[1][2] - f[1][0]) * (f[1][3] - f[1][1]))
        return kept


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area = lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1)
    return inter / (area(a) + area(b) - inter)


class Recognizer:
    """ArcFace w600k_r50: 512-d identity embedding of a 112x112 aligned face."""

    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=providers())
        self.input = self.sess.get_inputs()[0].name

    def embed(self, rgb_aligned_112):
        blob = ((rgb_aligned_112.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
        e = self.sess.run(None, {self.input: blob})[0][0]
        return e / np.linalg.norm(e)


class AgeGender:
    """insightface genderage: returns an age estimate in years (used only to leave out young-looking faces)."""

    def __init__(self, path):
        self.sess = ort.InferenceSession(path, providers=providers())
        self.input = self.sess.get_inputs()[0].name

    def age(self, rgb, box):
        w, h = box[2] - box[0], box[3] - box[1]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        scale = 96 / (max(w, h) * 1.5)
        m = np.array([[scale, 0, 48 - center[0] * scale], [0, scale, 48 - center[1] * scale]], np.float32)
        crop = cv2.warpAffine(rgb, m, (96, 96), borderValue=0).astype(np.float32)
        blob = crop.transpose(2, 0, 1)[None]
        out = self.sess.run(None, {self.input: blob})[0][0]
        return float(out[2] * 100)


def emap_from_swapper(onnx_path):
    """The identity projection matrix stored as the swap model's last initializer (512x512)."""
    import onnx

    model = onnx.load(onnx_path)
    from onnx import numpy_helper

    return numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)


def latent_from_embedding(emb, emap):
    lat = emb @ emap
    return lat / np.linalg.norm(lat, axis=-1, keepdims=True)
