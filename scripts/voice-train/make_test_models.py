"""Tiny stand-in voice models (a few KB) with the same inputs and outputs as the real ones.

The app's self-test uses them on Windows CI to check the whole AI voice path (engine process, ports,
audio worklet, the call window) without downloading the real models. The "voice" just hums at the
pitch it is given.

    python make_test_models.py ../../testdata/voice
"""
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = [helper.make_opsetid("", 17)]


def const(name, value, dtype=np.float32):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def content_model():
    # audio [1, N] -> feats [1, T, 768], T = floor((N - 400) / 320) + 1 (like ContentVec)
    nodes = [
        helper.make_node("Unsqueeze", ["audio", "ax1"], ["a3"]),
        helper.make_node("Conv", ["a3", "w"], ["c"], kernel_shape=[400], strides=[320]),
        helper.make_node("Transpose", ["c"], ["ct"], perm=[0, 2, 1]),
        helper.make_node("Mul", ["ct", "ones"], ["feats"]),
    ]
    inits = [const("ax1", [1], np.int64), const("w", np.full((1, 1, 400), 1 / 400)), const("ones", np.ones(768))]
    g = helper.make_graph(
        nodes, "content",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "N"])],
        [helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, "T", 768])], inits)
    return helper.make_model(g, opset_imports=OPSET, ir_version=8)


def pitch_model():
    # input [1, 128, T] -> output [1, T, 360]: always a clear peak at about 150 Hz
    bins = np.arange(360)
    peak = np.round((1200 * np.log2(150 / 10) - 1997.3794084376191) / 20)
    bias = 0.9 * np.exp(-0.5 * ((bins - peak) / 1.5) ** 2)
    nodes = [
        helper.make_node("ReduceMean", ["input"], ["m"], axes=[1], keepdims=1),
        helper.make_node("Transpose", ["m"], ["mt"], perm=[0, 2, 1]),
        helper.make_node("Mul", ["mt", "zeros"], ["z"]),
        helper.make_node("Add", ["z", "bias"], ["output"]),
    ]
    inits = [const("zeros", np.zeros(360)), const("bias", bias)]
    g = helper.make_graph(
        nodes, "pitch",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 128, "time"])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, "time", 360])], inits)
    return helper.make_model(g, opset_imports=OPSET, ir_version=8)


def voice_model(sr=40000):
    # Same inputs as an exported RVC voice. Output: a hum at nsff0 for the newest `rate` of the frames.
    hop = sr // 100
    nodes = [
        helper.make_node("Shape", ["nsff0"], ["shape"]),
        helper.make_node("Gather", ["shape", "one"], ["t"], axis=0),  # T (int64 scalar)
        helper.make_node("Cast", ["t"], ["tf"], to=TensorProto.FLOAT),
        helper.make_node("Sub", ["onef", "rate"], ["keep"]),
        helper.make_node("Mul", ["tf", "keep"], ["headf"]),
        helper.make_node("Floor", ["headf"], ["headfl"]),
        helper.make_node("Cast", ["headfl"], ["head"], to=TensorProto.INT64),
        helper.make_node("Slice", ["nsff0", "head", "big", "one1"], ["f0"]),  # [1, K]
        helper.make_node("Unsqueeze", ["f0", "ax2"], ["f03"]),  # [1, K, 1]
        helper.make_node("Expand", ["f03", "hopshape"], ["f0e"]),  # [1, K, hop]
        helper.make_node("Reshape", ["f0e", "flat"], ["f0s"]),  # [1, K*hop]
        helper.make_node("CumSum", ["f0s", "one0"], ["phase"]),
        helper.make_node("Mul", ["phase", "w"], ["ph"]),
        helper.make_node("Sin", ["ph"], ["s"]),
        helper.make_node("Mul", ["s", "amp"], ["s2"]),
        helper.make_node("Unsqueeze", ["s2", "ax1"], ["audio"]),
    ]
    inits = [
        const("one", 1, np.int64), const("onef", [1.0]), const("big", [1 << 30], np.int64), const("one1", [1], np.int64),
        const("ax2", [2], np.int64), const("ax1", [1], np.int64), const("hopshape", [1, 1, hop], np.int64),
        const("flat", [1, -1], np.int64), const("one0", 1, np.int64), const("w", 2 * np.pi / sr), const("amp", 0.3),
    ]
    inputs = [
        helper.make_tensor_value_info("phone", TensorProto.FLOAT, [1, "T", 768]),
        helper.make_tensor_value_info("phone_lengths", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("sid", TensorProto.INT64, [1]),
        helper.make_tensor_value_info("rate", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("pitch", TensorProto.INT64, [1, "T"]),
        helper.make_tensor_value_info("nsff0", TensorProto.FLOAT, [1, "T"]),
    ]
    g = helper.make_graph(nodes, "voice", inputs, [helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 1, "S"])], inits)
    return helper.make_model(g, opset_imports=OPSET, ir_version=8)


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for name, model in [("contentvec.onnx", content_model()), ("rmvpe.onnx", pitch_model()), ("voice-test.onnx", voice_model())]:
        onnx.checker.check_model(model)
        onnx.save(model, os.path.join(out, name))
        print(name, os.path.getsize(os.path.join(out, name)), "bytes")
