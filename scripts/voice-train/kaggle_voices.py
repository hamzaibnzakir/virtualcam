"""Train the four Brainbox AI voices on a free Kaggle GPU, in one run (about 5-7 hours on a T4).

No real person's voice is used. Each voice learns from speech made by Kokoro (an open text-to-speech
model, Apache-2.0) reading public-domain books, then RVC v2 (via Applio, MIT) is trained on it.

Steps (all logged to /kaggle/working/run.log, summed up in status.json):
  1. set up: Applio at a fixed commit, Kokoro, their downloads
  2. per voice: ~20 minutes of Kokoro speech -> Applio preprocess -> extract -> train -> export ONNX
  3. the two shared speech models (contentvec.onnx, rmvpe.onnx), and the three "My voices" models
     (wavlm6.onnx, knn-vocoder.onnx, knn-match.onnx: voices from uploads, no training; done first)
  4. sample clips (another Kokoro voice converted into each voice), measured pitch, sizes, fingerprints
  5. voice-models.json and my-voice-models.json: the manifests the app downloads from

Output, in /kaggle/working/:
  status.json, run.log
  upload/       everything to publish: voice-models.json, my-voice-models.json and every .onnx file
  samples/      before/after clips per voice (listen to these)
  datasets/     the generated speech (kept small: first 30 clips per voice)

Settings (environment variables): EPOCHS=200 MINUTES=20 BATCH=8 SESSION_HOURS=11.5 VOICES=all
  (VOICES can be e.g. "young-man,older-woman" to redo only some.)
"""
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

START = time.time()
ROOT = os.environ.get("KAGGLE_ROOT", "/kaggle")
W = os.path.join(ROOT, "working")
INPUT = os.path.join(ROOT, "input")
WORK = os.environ.get("WORK_DIR", "/tmp/voices")
LOCAL_TEST = os.environ.get("LOCAL_TEST") == "1"  # tiny CPU check of this script off Kaggle
EPOCHS = int(os.environ.get("EPOCHS", 2 if LOCAL_TEST else 200))
MINUTES = float(os.environ.get("MINUTES", 0.3 if LOCAL_TEST else 20))
BATCH = int(os.environ.get("BATCH", 2 if LOCAL_TEST else 8))
SESSION_HOURS = float(os.environ.get("SESSION_HOURS", 11.5))
SR = 40000
APPLIO_REPO = "https://github.com/IAHispano/Applio.git"
APPLIO_COMMIT = "445b0f6"
APPLIO = os.environ.get("APPLIO_DIR", os.path.join(WORK, "applio"))
KNNVC_REPO = "https://github.com/bshall/knn-vc.git"
KNNVC_COMMIT = "c616845"
KNNVC = os.environ.get("KNNVC_DIR", os.path.join(WORK, "knn-vc"))
RELEASE_BASE = "https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/download/models-voice"

# Kokoro voices used as teachers (computer-made voices; no real person's recordings).
VOICES = {
    "young-man": {"kokoro": "am_michael", "lang": "a", "speed": 1.0, "label": "Young man"},
    "young-woman": {"kokoro": "af_heart", "lang": "a", "speed": 1.0, "label": "Young woman"},
    "older-man": {"kokoro": "bm_george", "lang": "b", "speed": 0.92, "label": "Older man"},
    "older-woman": {"kokoro": "bf_emma", "lang": "b", "speed": 0.92, "label": "Older woman"},
}
# The "you" for the sample clips: a different Kokoro voice, converted into each trained voice.
SAMPLE_SPEAKERS = [("am_echo", "a"), ("af_sky", "a")]
SAMPLE_TEXT = "Hi, can you hear me okay? I'm trying out a new voice today. Let me know how it sounds on your end."

# Public-domain books (Project Gutenberg) for the voices to read.
BOOKS = [
    "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # The Adventures of Sherlock Holmes
    "https://www.gutenberg.org/cache/epub/11/pg11.txt",  # Alice's Adventures in Wonderland
    "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/cache/epub/84/pg84.txt",  # Frankenstein
]
FALLBACK_TEXT = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. It's easy to tell the depth of a well. "
    "These days a chicken leg is a rare dish. Rice is often served in round bowls. The juice of lemons makes fine punch. "
    "The box was thrown beside the parked truck. The hogs were fed chopped corn and garbage. Four hours of steady work faced us. "
    "A large size in stockings is hard to sell. The boy was there when the sun rose. A rod is used to catch pink salmon. "
    "The source of the huge river is the clear spring. Kick the ball straight and follow through. Help the woman get back to her feet. "
    "A pot of tea helps to pass the evening. Smoky fires lack flame and heat. The soft cushion broke the man's fall. "
    "The salt breeze came across from the sea. The girl at the booth sold fifty bonds. The small pup gnawed a hole in the sock. "
    "The fish twisted and turned on the bent hook. Press the pants and sew a button on the vest. The swan dive was far short of perfect. "
    "What time does the meeting start tomorrow? I honestly didn't expect that, but it's great news! Could you send me the file again, please? "
    "We should probably leave before the traffic gets bad. Thanks so much for calling, it was really good to hear from you."
)

os.makedirs(W, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
status = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "settings": {"epochs": EPOCHS, "minutes": MINUTES, "batch": BATCH, "local_test": LOCAL_TEST}, "stages": [], "voices": {}}
log_file = open(os.path.join(W, "run.log"), "a")


def say(msg):
    line = f'[{time.strftime("%H:%M:%S")} +{(time.time() - START) / 3600:.2f}h] {msg}'
    print(line, flush=True)
    log_file.write(line + "\n")
    log_file.flush()


def save_status():
    status["hours"] = round((time.time() - START) / 3600, 2)
    with open(os.path.join(W, "status.json"), "w") as f:
        json.dump(status, f, indent=2)


def stage(name, ok, **info):
    status["stages"].append({"stage": name, "ok": ok, "hours": round((time.time() - START) / 3600, 2), **info})
    save_status()
    say(f"{'OK' if ok else 'FAILED'}: {name} {json.dumps(info)[:400]}")


def run(cmd, cwd=None, timeout=None, check=True):
    say("$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    deadline = time.time() + timeout if timeout else None
    for line in p.stdout:
        line = line.rstrip()
        lines.append(line)
        log_file.write(line + "\n")
        # progress lines are long and frequent; print a sample
        if len(lines) % 20 == 1 or "rror" in line or "poch" in line[:40]:
            print(line[:300], flush=True)
        if deadline and time.time() > deadline:
            p.kill()
            say(f"stopped after {timeout / 3600:.2f} h (time budget)")
            break
    code = p.wait()
    log_file.flush()
    if check and code not in (0, -9):
        raise RuntimeError(f"command failed ({code}): {' '.join(cmd)}\n" + "\n".join(lines[-30:]))
    return code, lines


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hours_left():
    return SESSION_HOURS - (time.time() - START) / 3600


def kit_dir():
    """This kit: next to this file, or mounted as a Kaggle dataset."""
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.exists(os.path.join(here, "convert_file.py")):
        return here
    for p in glob.glob(os.path.join(INPUT, "**", "convert_file.py"), recursive=True):
        return os.path.dirname(p)
    raise RuntimeError("Couldn't find the kit (convert_file.py) next to this script or in /kaggle/input")


# ---------- 1. set up ----------

def setup():
    if LOCAL_TEST:
        stage("setup", True, note="local test: using the existing Applio folder and Python packages")
        return
    run(["apt-get", "install", "-y", "-qq", "espeak-ng"], check=False)
    if not os.path.exists(os.path.join(APPLIO, "core.py")):
        run(["git", "clone", "-q", APPLIO_REPO, APPLIO])
    run(["git", "checkout", "-q", APPLIO_COMMIT], cwd=APPLIO)
    # Kaggle already has a CUDA build of torch: keep it, install the rest of what training needs.
    pkgs = ["kokoro>=0.9.4", "misaki[en]>=0.9.4", "soundfile", "faiss-cpu", "soxr", "einops", "torchcrepe", "torchfcpe",
            "stftpitchshift", "onnx", "onnxslim", "onnxruntime", "librosa", "click", "tensorboard", "local-attention", "transformers>=4.44"]
    run([sys.executable, "-m", "pip", "install", "-q", *pkgs])
    run([sys.executable, "core.py", "prerequisites", "--pretraineds-hifigan", "--models", "--no-exe"], cwd=APPLIO)
    import torch

    stage("setup", True, applio=APPLIO_COMMIT, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


# ---------- 2. speech for each voice ----------

def corpus():
    texts = []
    if not LOCAL_TEST:
        for url in BOOKS:
            try:
                raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "ignore")
                start = raw.find("*** START OF")
                end = raw.find("*** END OF")
                body = raw[raw.find("\n", start) + 1:end] if start >= 0 and end > start else raw
                texts.append(body)
            except Exception as err:  # noqa: BLE001
                say(f"couldn't fetch {url}: {err}")
    texts.append(FALLBACK_TEXT)
    sentences = []
    for t in texts:
        t = re.sub(r"\s+", " ", t.replace("_", " "))
        for s in re.split(r"(?<=[.!?])\s+", t):
            s = s.strip()
            if 25 <= len(s) <= 220 and not re.search(r"[\[\]{}<>#*@|]|CHAPTER|http", s) and s.upper() != s:
                sentences.append(s)
    # Mix the books so every voice reads a bit of each.
    import random

    random.Random(7).shuffle(sentences)
    return sentences


_pipelines = {}


def tts(text, voice, lang, speed):
    """Kokoro speech at 24 kHz (float32 numpy). The local test uses espeak instead."""
    import numpy as np

    if LOCAL_TEST:
        import librosa

        tmp = os.path.join(WORK, "espeak.wav")
        subprocess.run(["espeak-ng", "-v", "en-us+f3" if "f_" in voice else "en-us+m3", "-w", tmp, text], check=True)
        a, _ = librosa.load(tmp, sr=24000)
        return a
    from kokoro import KPipeline

    if lang not in _pipelines:
        _pipelines[lang] = KPipeline(lang_code=lang)
    parts = [r.audio.cpu().numpy() if hasattr(r.audio, "cpu") else np.asarray(r.audio) for r in _pipelines[lang](text, voice=voice, speed=speed) if r.audio is not None]
    return np.concatenate(parts) if parts else np.zeros(0, np.float32)


def make_dataset(vid, cfg, sentences, offset):
    import numpy as np
    import soundfile as sf

    out = os.path.join(WORK, "datasets", vid)
    if os.path.isdir(out) and len(os.listdir(out)) > 5:
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    total = 0.0
    n = 0
    i = offset
    while total < MINUTES * 60 and i < offset + len(sentences):
        # A few sentences per clip (5-15 s): natural flow, and Applio cuts them into 3 s pieces.
        text = " ".join(sentences[(i + k) % len(sentences)] for k in range(3))
        i += 3
        audio = tts(text, cfg["kokoro"], cfg["lang"], cfg["speed"])
        if len(audio) < 24000:
            continue
        peak = float(np.abs(audio).max()) or 1.0
        sf.write(os.path.join(out, f"{n:04d}.wav"), (audio / peak * 0.8).astype(np.float32), 24000)
        total += len(audio) / 24000
        n += 1
    # Keep a small copy with the run's output, to listen to.
    keep = os.path.join(W, "datasets", vid)
    os.makedirs(keep, exist_ok=True)
    for f in sorted(os.listdir(out))[:30]:
        shutil.copy(os.path.join(out, f), keep)
    return out, total, n


def speech_pitch(folder, pitch_session):
    """Median speaking pitch (Hz) of a voice's dataset: the app moves your pitch to this."""
    import librosa
    import numpy as np

    sys.path.insert(0, kit_dir())
    from convert_file import pitch

    values = []
    for f in sorted(glob.glob(os.path.join(folder, "*.wav")))[:40]:
        a, _ = librosa.load(f, sr=16000)
        f0 = pitch(pitch_session, a)
        values.extend(f0[(f0 > 60) & (f0 < 600)].tolist())
    return float(np.median(values)) if values else 0.0


# ---------- 3. train one voice ----------

def latest_weights(name):
    logs = os.path.join(os.environ.get("APPLIO_LOGS_DIR") or os.path.join(APPLIO, "logs"), name)
    found = []
    for p in glob.glob(os.path.join(logs, f"{name}_*e_*s.pth")):
        m = re.search(r"_(\d+)e_(\d+)s\.pth$", p)
        if m:
            found.append((int(m.group(2)), p))
    return max(found)[1] if found else None


def train_voice(vid, dataset, budget_hours):
    name = f"brainbox-{vid}"
    py = sys.executable
    gpu = "-" if LOCAL_TEST else "0"
    cores = str(min(4, os.cpu_count() or 2))
    run([py, "core.py", "preprocess", "--model-name", name, "--dataset-path", dataset, "--sample-rate", str(SR), "--cpu-cores", cores,
         "--cut-preprocess", "Automatic", "--chunk-len", "3.0", "--overlap-len", "0.3", "--normalization-mode", "post"], cwd=APPLIO)
    run([py, "core.py", "extract", "--model-name", name, "--f0-method", "rmvpe", "--cpu-cores", cores, "--gpu", gpu,
         "--sample-rate", str(SR), "--embedder-model", "contentvec", "--include-mutes", "2"], cwd=APPLIO)
    save_every = max(1, min(25, EPOCHS // 8))
    train_cmd = [py, "core.py", "train", "--model-name", name, "--save-every-epoch", str(save_every), "--save-only-latest",
                 "--total-epoch", str(EPOCHS), "--sample-rate", str(SR), "--batch-size", str(BATCH), "--gpu", gpu,
                 "--pretrained", "--index-algorithm", "Auto"]
    if not LOCAL_TEST:
        train_cmd.append("--save-every-weights")
    t = time.time()
    run(train_cmd, cwd=APPLIO, timeout=max(600, budget_hours * 3600), check=False)
    weights = latest_weights(name)
    if not weights:
        raise RuntimeError(f"training {vid} produced no weights (see run.log)")
    return weights, round((time.time() - t) / 3600, 2)


def export_voice(vid, weights, upload):
    import onnx

    sys.path.insert(0, kit_dir())
    from half_storage import half_storage

    full = os.path.join(WORK, f"voice-{vid}.full.onnx")
    run([sys.executable, "core.py", "model-onnx-export", "--pth-path", weights, "--output-path", full], cwd=APPLIO, check=False)
    if not os.path.exists(full):
        # Older click versions name the command with an underscore.
        run([sys.executable, "core.py", "model_onnx_export", "--pth-path", weights, "--output-path", full], cwd=APPLIO)
    out = os.path.join(upload, f"voice-{vid}.onnx")
    half_storage(full, out)
    m = onnx.load(out)
    if m.ir_version > 10:
        m.ir_version = 10
        onnx.save(m, out)
    os.remove(full)
    return out


# ---------- main ----------

def main():
    setup()
    import onnxruntime as ort

    kit = kit_dir()
    sys.path.insert(0, kit)
    upload = os.path.join(W, "upload")
    os.makedirs(upload, exist_ok=True)
    # A previous run attached as an input (kernel_sources): keep everything it made.
    for prev in glob.glob(os.path.join(INPUT, "**", "upload", "voice-models.json"), recursive=True):
        src = os.path.dirname(prev)
        for f in os.listdir(src):
            if not os.path.exists(os.path.join(upload, f)):
                shutil.copy(os.path.join(src, f), upload)
        stage("previous run", True, folder=src, files=sorted(os.listdir(src)))

    # "My voices" models first: they need no training, so they're ready even if a voice fails later.
    try:
        if not all(os.path.exists(os.path.join(upload, f)) for f in ("wavlm6.onnx", "knn-vocoder.onnx", "knn-match.onnx")):
            if not os.path.exists(os.path.join(KNNVC, "matcher.py")):
                run(["git", "clone", "-q", KNNVC_REPO, KNNVC])
            run(["git", "checkout", "-q", KNNVC_COMMIT], cwd=KNNVC)
            run([sys.executable, os.path.join(kit, "export_clone.py"), "--knnvc", KNNVC, "--out", upload, "--cache", WORK])
        files = []
        for f in ("wavlm6.onnx", "knn-vocoder.onnx", "knn-match.onnx"):
            p = os.path.join(upload, f)
            files.append({"file": f, "label": {"wavlm6.onnx": "Speech reader", "knn-vocoder.onnx": "Voice maker", "knn-match.onnx": "Voice matcher"}[f],
                          "url": f"{RELEASE_BASE}/{f}", "sha256": sha256(p), "bytes": os.path.getsize(p)})
        with open(os.path.join(upload, "my-voice-models.json"), "w") as fh:
            json.dump({"version": int(time.strftime("%Y%m%d")), "files": files}, fh, indent=2)
        stage("my voice models", True, mb={f["file"]: round(f["bytes"] / 1e6, 1) for f in files})
    except Exception as err:  # noqa: BLE001
        stage("my voice models", False, error=str(err)[-1500:])

    # Shared models (needed to measure pitch and make samples).
    shared_ok = all(os.path.exists(os.path.join(upload, f)) for f in ("contentvec.onnx", "rmvpe.onnx"))
    if not shared_ok:
        run([sys.executable, os.path.join(kit, "export_shared.py"), "--contentvec-dir", os.path.join(APPLIO, "rvc", "models", "embedders", "contentvec"), "--out", upload])
    stage("shared models", True, files=["contentvec.onnx", "rmvpe.onnx"])
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    content = ort.InferenceSession(os.path.join(upload, "contentvec.onnx"), providers=providers)
    pitch_model = ort.InferenceSession(os.path.join(upload, "rmvpe.onnx"), providers=providers)

    wanted = [v for v in os.environ.get("VOICES", "all").split(",") if v in VOICES] or list(VOICES)
    if LOCAL_TEST:
        wanted = wanted[:1]
    sentences = corpus()
    stage("text", True, sentences=len(sentences))
    manifest_path = os.path.join(upload, "voice-models.json")
    voices = {}
    if os.path.exists(manifest_path):
        voices = json.load(open(manifest_path)).get("voices", {})

    for n, vid in enumerate(wanted):
        cfg = VOICES[vid]
        left = hours_left()
        remaining = len(wanted) - n
        budget = max(0.2, (left - 0.4) / remaining - 0.35)  # leave time for speech, export and samples
        if left < 0.7:
            stage(f"voice {vid}", False, reason="out of session time; run again with VOICES set to the missing ones")
            continue
        try:
            t = time.time()
            dataset, seconds, clips = make_dataset(vid, cfg, sentences, offset=n * 997)
            f0 = speech_pitch(dataset, pitch_model)
            say(f"{vid}: {seconds / 60:.1f} min of speech in {clips} clips, median pitch {f0:.0f} Hz ({(time.time() - t) / 60:.1f} min)")
            weights, train_hours = train_voice(vid, dataset, budget)
            onnx_path = export_voice(vid, weights, upload)
            voices[vid] = {"file": os.path.basename(onnx_path), "sr": SR, "f0": round(f0, 1)}
            status["voices"][vid] = {"teacher": cfg["kokoro"], "speech_minutes": round(seconds / 60, 1), "clips": clips, "pitch_hz": round(f0, 1),
                                     "weights": os.path.basename(weights), "train_hours": train_hours, "onnx_mb": round(os.path.getsize(onnx_path) / 1e6, 1)}
            stage(f"voice {vid}", True, **status["voices"][vid])
        except Exception as err:  # noqa: BLE001
            stage(f"voice {vid}", False, error=str(err)[-1500:])

    # Samples: another computer voice, converted into each trained voice.
    import numpy as np
    import soundfile as sf
    from convert_file import convert

    samples = os.path.join(W, "samples")
    os.makedirs(samples, exist_ok=True)
    import librosa

    sources = []
    for spk, lang in SAMPLE_SPEAKERS[: 1 if LOCAL_TEST else 2]:
        a24 = tts(SAMPLE_TEXT, spk, lang, 1.0)
        src = os.path.join(samples, f"input-{spk}.wav")
        sf.write(src, a24, 24000)
        sources.append((spk, librosa.resample(a24, orig_sr=24000, target_sr=16000)))
    for vid, v in voices.items():
        sess = ort.InferenceSession(os.path.join(upload, v["file"]), providers=providers)
        for spk, a16 in sources:
            out, info = convert(sess, content, pitch_model, a16.astype(np.float32), v["f0"])
            sf.write(os.path.join(samples, f"{vid}-from-{spk}.wav"), out, SR)
            status["voices"].setdefault(vid, {})[f"sample_from_{spk}"] = {"shift_semitones": round(info["shift"], 1), "seconds": round(len(out) / SR, 1)}
    stage("samples", True, files=sorted(os.listdir(samples)))

    # Manifest for the app (files hosted on the releases repo under the tag models-voice).
    labels = {"contentvec.onnx": "Speech features", "rmvpe.onnx": "Pitch finder"}
    files = []
    for f in ["contentvec.onnx", "rmvpe.onnx"] + sorted(v["file"] for v in voices.values()):
        p = os.path.join(upload, f)
        vid = next((k for k, v in voices.items() if v["file"] == f), None)
        files.append({"file": f, "label": labels.get(f, f"AI voice: {VOICES[vid]['label'].lower()}" if vid else f), "url": f"{RELEASE_BASE}/{f}", "sha256": sha256(p), "bytes": os.path.getsize(p)})
    manifest = {"version": int(time.strftime("%Y%m%d")), "files": files, "voices": voices}
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    status["upload"] = {f["file"]: f["bytes"] for f in files}
    status["done"] = len(voices) == len(VOICES)
    stage("manifest", True, voices=sorted(voices), total_mb=round(sum(f["bytes"] for f in files) / 1e6))


if __name__ == "__main__":
    try:
        main()
    except Exception as err:  # noqa: BLE001
        stage("run", False, error=str(err)[-3000:])
        raise
    finally:
        save_status()
