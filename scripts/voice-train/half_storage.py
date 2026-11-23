"""Halve an ONNX model's download size without changing how it runs.

Every large float32 weight is stored as float16 and turned back into float32 by a Cast node when the
model loads (onnxruntime folds the casts away), so the model still computes in float32 and runs on
any device, including plain CPUs (a true float16 model doesn't run on every CPU).

    python half_storage.py in.onnx out.onnx
"""
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

MIN_ELEMENTS = 1024  # small tensors (shapes, scalars) stay as they are


def half_storage(src, dst):
    model = onnx.load(src)
    graph = model.graph
    keep, casts = [], []
    for init in graph.initializer:
        if init.data_type != onnx.TensorProto.FLOAT or np.prod(init.dims) < MIN_ELEMENTS:
            keep.append(init)
            continue
        arr = numpy_helper.to_array(init)
        if np.abs(arr).max() > 60000:  # would overflow float16
            keep.append(init)
            continue
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT, name=init.name + "__cast"))
    del graph.initializer[:]
    graph.initializer.extend(keep)
    nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(casts + nodes)
    onnx.save(model, dst)
    return len(casts)


if __name__ == "__main__":
    n = half_storage(sys.argv[1], sys.argv[2])
    print(f"stored {n} weights as float16 -> {sys.argv[2]}")
