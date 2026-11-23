"""Make the two speech models every AI voice shares, in the form the app loads.

  contentvec.onnx  "what is being said" features (ContentVec, the same weights Applio trains with)
  rmvpe.onnx       pitch finder (RMVPE, from the RVC project)

Both are stored at half size (float16 weights, float32 maths; see half_storage.py) and checked against
the originals.

    python export_shared.py --contentvec-dir <dir with config.json + pytorch_model.bin> --out <dir>
"""
import argparse
import os
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
import torch

from half_storage import half_storage

RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx"


class Content(torch.nn.Module):
    def __init__(self, hubert):
        super().__init__()
        self.hubert = hubert

    def forward(self, audio):
        return self.hubert(audio).last_hidden_state


def export_contentvec(src_dir, out):
    from transformers import HubertModel

    model = HubertModel.from_pretrained(src_dir).eval()
    wrapped = Content(model).eval()
    full = out + ".full.onnx"
    x = torch.randn(1, 16000) * 0.1
    with torch.no_grad():
        torch.onnx.export(wrapped, (x,), full, input_names=["audio"], output_names=["feats"],
                          dynamic_axes={"audio": {1: "N"}, "feats": {1: "T"}}, opset_version=17, dynamo=False)
    # Check on a fresh input (the export must not have changed what the model does).
    test = torch.from_numpy(np.random.RandomState(1).randn(1, 20960).astype(np.float32) * 0.1)
    with torch.no_grad():
        ref = wrapped(test).numpy()
    half_storage(full, out)
    os.remove(full)
    got = ort.InferenceSession(out).run(None, {"audio": test.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    return {"file": os.path.basename(out), "max_diff": diff, "frames": int(got.shape[1])}


def export_rmvpe(out):
    full = out + ".full.onnx"
    urllib.request.urlretrieve(RMVPE_URL, full)
    mel = np.random.RandomState(2).uniform(-8, 0, (1, 128, 160)).astype(np.float32)
    ref = ort.InferenceSession(full).run(None, {"input": mel})[0]
    half_storage(full, out)
    os.remove(full)
    got = ort.InferenceSession(out).run(None, {"input": mel})[0]
    return {"file": os.path.basename(out), "max_diff": float(np.abs(got - ref).max())}


def fix_ir(path, max_ir=10):
    """The app's runtime reads model format (IR) version 10 or older."""
    m = onnx.load(path)
    if m.ir_version > max_ir:
        m.ir_version = max_ir
        onnx.save(m, path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--contentvec-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = export_contentvec(a.contentvec_dir, os.path.join(a.out, "contentvec.onnx"))
    r = export_rmvpe(os.path.join(a.out, "rmvpe.onnx"))
    for f in ("contentvec.onnx", "rmvpe.onnx"):
        fix_ir(os.path.join(a.out, f))
    print({"contentvec": c, "rmvpe": r})
    assert c["max_diff"] < 0.02 and r["max_diff"] < 0.01, "a shared model doesn't match its original"
