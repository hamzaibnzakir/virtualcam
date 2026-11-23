"""Make the models for "My voices" (a voice from your own upload, ready in about a minute, no training).

Method: kNN-VC (Baas, van Niekerk, Kamper, 2023; code MIT, https://github.com/bshall/knn-vc). Every
20 ms of your speech is described by WavLM (Microsoft, MIT) and replaced by the average of the most
similar moments in the uploaded voice; a vocoder turns that back into sound.

  wavlm6.onnx       audio 16 kHz [1, N] -> feats [1, T, 1024]   (WavLM-Large, first 6 layers only)
  knn-vocoder.onnx  feats [1, T, 1024]  -> audio [1, 1, T*320]  (HiFi-GAN trained on prematched feats)
  knn-match.onnx    query [Q, 1024], set [S, 1024] -> matched [Q, 1024] (cosine, mean of the 4 nearest)

Weights are stored at half size (half_storage.py) and every model is checked against PyTorch.

    python export_clone.py --knnvc <clone of bshall/knn-vc> --out <dir>
"""
import argparse
import json
import os
import sys
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from half_storage import half_storage

WAVLM_URL = "https://github.com/bshall/knn-vc/releases/download/v0.1/WavLM-Large.pt"
VOCODER_URL = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
LAYERS = 6  # kNN-VC uses the output of layer 6
TOPK = 4


def fetch(url, cache):
    path = os.path.join(cache, os.path.basename(url))
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


class Features(torch.nn.Module):
    def __init__(self, wavlm):
        super().__init__()
        self.wavlm = wavlm

    def forward(self, audio):
        return self.wavlm.extract_features(audio, output_layer=LAYERS, ret_layer_results=False)[0]


class Vocoder(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.g = g

    def forward(self, feats):
        return self.g(feats)


def match_model():
    """Cosine kNN: for each query row, the mean of the TOPK closest rows of `set`."""
    eps = helper.make_tensor("eps", TensorProto.FLOAT, [], [1e-8])
    k = helper.make_tensor("k", TensorProto.INT64, [1], [TOPK])
    nodes = [
        helper.make_node("ReduceL2", ["query"], ["qn"], axes=[1], keepdims=1),
        helper.make_node("Add", ["qn", "eps"], ["qn2"]),
        helper.make_node("Div", ["query", "qn2"], ["q"]),
        helper.make_node("ReduceL2", ["set"], ["sn"], axes=[1], keepdims=1),
        helper.make_node("Add", ["sn", "eps"], ["sn2"]),
        helper.make_node("Div", ["set", "sn2"], ["s"]),
        helper.make_node("Transpose", ["s"], ["st"], perm=[1, 0]),
        helper.make_node("MatMul", ["q", "st"], ["sim"]),  # [Q, S]
        helper.make_node("TopK", ["sim", "k"], ["vals", "idx"], axis=1, largest=1),  # [Q, K]
        helper.make_node("Gather", ["set", "idx"], ["picked"], axis=0),  # [Q, K, 1024] (the raw rows)
        helper.make_node("ReduceMean", ["picked"], ["matched"], axes=[1], keepdims=0),
    ]
    g = helper.make_graph(
        nodes, "knn-match",
        [helper.make_tensor_value_info("query", TensorProto.FLOAT, ["Q", 1024]), helper.make_tensor_value_info("set", TensorProto.FLOAT, ["S", 1024])],
        [helper.make_tensor_value_info("matched", TensorProto.FLOAT, ["Q", 1024])], [eps, k])
    return helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def check(path, feeds, ref, tol):
    got = ort.InferenceSession(path).run(None, feeds)[0]
    diff = float(np.abs(got.reshape(ref.shape) - ref).max())
    scale = float(np.abs(ref).max())
    assert diff <= tol * max(1.0, scale), f"{os.path.basename(path)} differs from PyTorch by {diff} (scale {scale})"
    return diff


def fix_ir(path, max_ir=10):
    m = onnx.load(path)
    if m.ir_version > max_ir:
        m.ir_version = max_ir
        onnx.save(m, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knnvc", required=True, help="folder of a clone of https://github.com/bshall/knn-vc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    sys.path.insert(0, a.knnvc)
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    from wavlm.WavLM import WavLM, WavLMConfig

    os.makedirs(a.out, exist_ok=True)
    cache = a.cache or a.out
    torch.manual_seed(0)
    report = {}

    # WavLM, cut to the layers kNN-VC uses.
    ck = torch.load(fetch(WAVLM_URL, cache), map_location="cpu", weights_only=False)
    wavlm = WavLM(WavLMConfig(ck["cfg"]))
    wavlm.load_state_dict(ck["model"])
    wavlm.encoder.layers = wavlm.encoder.layers[:LAYERS]
    wavlm.eval()
    feats_model = Features(wavlm).eval()
    full = os.path.join(a.out, "wavlm6.full.onnx")
    with torch.no_grad():
        torch.onnx.export(feats_model, (torch.randn(1, 16000) * 0.1,), full, input_names=["audio"], output_names=["feats"],
                          dynamic_axes={"audio": {1: "N"}, "feats": {1: "T"}}, opset_version=17, dynamo=False)
        test = torch.from_numpy(np.random.RandomState(1).randn(1, 24000).astype(np.float32) * 0.1)
        ref = feats_model(test).numpy()
    out = os.path.join(a.out, "wavlm6.onnx")
    half_storage(full, out)
    os.remove(full)
    report["wavlm6"] = check(out, {"audio": test.numpy()}, ref, 0.02)

    # Vocoder.
    h = AttrDict(json.load(open(os.path.join(a.knnvc, "hifigan", "config_v1_wavlm.json"))))
    g = Generator(h)
    g.load_state_dict(torch.load(fetch(VOCODER_URL, cache), map_location="cpu")["generator"])
    g.eval()
    g.remove_weight_norm()
    voc = Vocoder(g).eval()
    full = os.path.join(a.out, "knn-vocoder.full.onnx")
    with torch.no_grad():
        torch.onnx.export(voc, (torch.from_numpy(ref[:, :20]),), full, input_names=["feats"], output_names=["audio"],
                          dynamic_axes={"feats": {1: "T"}, "audio": {2: "S"}}, opset_version=17, dynamo=False)
        vref = voc(torch.from_numpy(ref)).numpy()
    out = os.path.join(a.out, "knn-vocoder.onnx")
    half_storage(full, out)
    os.remove(full)
    report["vocoder"] = check(out, {"feats": ref}, vref, 0.02)

    # Matcher (no weights).
    out = os.path.join(a.out, "knn-match.onnx")
    onnx.save(match_model(), out)
    q = np.random.RandomState(3).randn(7, 1024).astype(np.float32)
    s = np.random.RandomState(4).randn(300, 1024).astype(np.float32)
    qn = q / np.linalg.norm(q, axis=1, keepdims=True)
    sn = s / np.linalg.norm(s, axis=1, keepdims=True)
    idx = np.argsort(-(qn @ sn.T), axis=1)[:, :TOPK]
    report["match"] = check(out, {"query": q, "set": s}, s[idx].mean(axis=1), 1e-4)

    for f in ("wavlm6.onnx", "knn-vocoder.onnx", "knn-match.onnx"):
        fix_ir(os.path.join(a.out, f))
        report[f + "_mb"] = round(os.path.getsize(os.path.join(a.out, f)) / 1e6, 1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
