"""Convert a recording with a trained voice, the same way the app does (whole file at once).

Used to make the sample clips for the training report, and as the reference the app's engine is
checked against.

    python convert_file.py --voice young-man.onnx --content contentvec.onnx --pitch rmvpe.onnx \
        --input speech.wav --output out.wav [--f0 120]

--f0: the voice's typical pitch; your pitch is moved to it (0 = keep your pitch).
"""
import argparse
import json

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 16000
CENTS = 20 * np.arange(360) + 1997.3794084376191
MEL = librosa.filters.mel(sr=SR, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)


def log_mel(audio):
    spec = np.abs(librosa.stft(audio, n_fft=1024, hop_length=160, win_length=1024, window="hann", center=True, pad_mode="reflect"))
    return np.log(np.clip(MEL @ spec, 1e-5, None)).astype(np.float32)


def pitch(session, audio, threshold=0.03):
    mel = log_mel(audio)
    frames = mel.shape[1]
    padded = 32 * ((frames - 1) // 32 + 1)
    mel = np.pad(mel, ((0, 0), (0, padded - frames)))
    sal = session.run(None, {"input": mel[None]})[0].reshape(-1, 360)[:frames]
    f0 = np.zeros(frames, np.float32)
    for t in range(frames):
        c = int(np.argmax(sal[t]))
        if sal[t, c] <= threshold:
            continue
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = sal[t, lo:hi]
        f0[t] = 10 * 2 ** ((w @ CENTS[lo:hi]) / w.sum() / 1200)
    return f0


def coarse(f0):
    lo, hi = 1127 * np.log(1 + 50 / 700), 1127 * np.log(1 + 1100 / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    mel[mel > 0] = (mel[mel > 0] - lo) * 254 / (hi - lo) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64)


def convert(voice, content, pitch_model, audio, target_f0=0.0, speaker_f0=None):
    p_len = len(audio) // 160
    feats = content.run(None, {"audio": audio[None].astype(np.float32)})[0][0]
    idx = np.minimum(np.arange(p_len) // 2, len(feats) - 1)
    phone = feats[idx][None].astype(np.float32)
    f0 = pitch(pitch_model, audio)[:p_len]
    voiced = f0[(f0 > 60) & (f0 < 500)]
    speaker_f0 = speaker_f0 or (float(np.median(voiced)) if len(voiced) else 0)
    shift = 0.0
    if target_f0 and speaker_f0:
        shift = float(np.clip(12 * np.log2(target_f0 / speaker_f0), -12, 12))
    nsf = (f0 * 2 ** (shift / 12)).astype(np.float32)
    out = voice.run(None, {
        "phone": phone,
        "phone_lengths": np.array([p_len], np.int64),
        "sid": np.array([0], np.int64),
        "rate": np.array([1.0], np.float32),
        "pitch": coarse(nsf)[None],
        "nsff0": nsf[None],
    })[0].reshape(-1)
    return out, {"speaker_f0": speaker_f0, "shift": shift}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--content", required=True)
    ap.add_argument("--pitch", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--f0", type=float, default=0.0)
    ap.add_argument("--sr", type=int, default=40000)
    a = ap.parse_args()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    mk = lambda f: ort.InferenceSession(f, providers=providers)
    audio, _ = librosa.load(a.input, sr=SR, mono=True)
    out, info = convert(mk(a.voice), mk(a.content), mk(a.pitch), audio, a.f0)
    sf.write(a.output, out, a.sr)
    print(json.dumps({"output": a.output, "seconds": round(len(out) / a.sr, 2), **info}))


if __name__ == "__main__":
    main()
