"""Exports the trained student to ONNX files the VirtualCam app can load in place of inswapper_128:
  <out>/inswapper_fast_<width>.onnx       float32
  <out>/inswapper_fast_<width>_fp16.onnx  float16 weights, float32 inputs/outputs (like inswapper_128_fp16)
Inputs `target` (1,3,128,128) and `source` (1,512), output `output` (1,3,128,128): the same as the teacher.
The teacher's identity matrix (emap) is stored as the files' last initializer, where the app looks for it.

  python export.py --ckpt /runs/fast1/ckpt_best.pt --models /models --out /runs/fast1/export
"""
import argparse
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

sys.path.insert(0, os.path.dirname(__file__))
from faces import emap_from_swapper  # noqa: E402
from student import Student  # noqa: E402


def add_emap(model, emap):
    model.graph.initializer.append(numpy_helper.from_array(emap.astype(np.float32), name='emap'))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location='cpu')
    width, blocks = ck['width'], ck['blocks']
    model = Student(width=width, blocks=blocks).eval()
    model.load_state_dict(ck['ema'])
    emap = emap_from_swapper(os.path.join(args.models, 'inswapper_128.onnx'))

    f32 = os.path.join(args.out, f'inswapper_fast_{width}.onnx')
    target = torch.rand(1, 3, 128, 128)
    source = torch.nn.functional.normalize(torch.randn(1, 512), dim=1)
    torch.onnx.export(model, (target, source), f32, input_names=['target', 'source'], output_names=['output'],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    m = onnx.load(f32)
    onnx.save(add_emap(m, emap), f32)

    from onnxconverter_common import float16

    f16 = os.path.join(args.out, f'inswapper_fast_{width}_fp16.onnx')
    m16 = float16.convert_float_to_float16(onnx.load(f32), keep_io_types=True)
    # The emap stays float32 (the app reads either); drop the converted copy and add it back.
    keep = [i for i in m16.graph.initializer if i.name != 'emap']
    del m16.graph.initializer[:]
    m16.graph.initializer.extend(keep)
    onnx.save(add_emap(m16, emap), f16)

    with torch.no_grad():
        ref = model(target, source).numpy()
    for path in (f32, f16):
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'target': target.numpy(), 'source': source.numpy()})[0]
        print(f'{os.path.basename(path)}: {os.path.getsize(path) / 1e6:.1f} MB, max diff vs PyTorch {np.abs(out - ref).max():.4f}')


if __name__ == '__main__':
    main()
