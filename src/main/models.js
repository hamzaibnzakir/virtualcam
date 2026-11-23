'use strict';

// Downloads the local AI models with progress, and checks every file against its known fingerprint.
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { Readable } = require('node:stream');

const MODELS = [
  {
    id: 'detector',
    file: 'det_10g.onnx',
    label: 'Face finder',
    url: 'https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx',
    sha256: '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91',
    bytes: 16923827,
    pack: 'core',
  },
  {
    id: 'recognizer',
    file: 'w600k_r50.onnx',
    label: 'Face identity',
    url: 'https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
    sha256: '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43',
    bytes: 174383860,
    pack: 'core',
  },
  {
    id: 'swapper',
    file: 'inswapper_128_fp16.onnx',
    label: 'Face swap',
    url: 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx',
    sha256: '6d51a9278a1f650cffefc18ba53f38bf2769bf4bbff89267822cf72945f8a38b',
    bytes: 277680638,
    pack: 'core',
  },
  {
    id: 'enhancer',
    file: 'GFPGANv1.4.onnx',
    label: 'Face detail boost',
    url: 'https://huggingface.co/hacksider/deep-live-cam/resolve/main/GFPGANv1.4.onnx',
    sha256: 'cd7311b8d9e13cdb1e208b12363182da58c7bf45e26d1aa67bbeac4751aae92e',
    bytes: 340256686,
    pack: 'boost',
  },
];

// The fast swap model (a small copy of the swap model, trained for this app) is published with the
// app's releases. Its details live in a small manifest next to it, so a newer model can be published
// without a new app version.
const FAST_MANIFEST_URL =
  'https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/download/models-fast/fast-model.json';
const FAST_LOCAL = 'fast-model.json';

function validFast(m) {
  return (
    m &&
    typeof m.file === 'string' &&
    /^[\w.-]+\.onnx$/.test(m.file) &&
    typeof m.url === 'string' &&
    /^https:\/\/(github\.com|huggingface\.co|objects\.githubusercontent\.com)\//.test(m.url) &&
    /^[0-9a-f]{64}$/.test(m.sha256) &&
    Number.isInteger(m.bytes) &&
    m.bytes > 1e6 &&
    m.bytes < 2e8
  );
}

function fastLocal(modelsDir) {
  try {
    const m = JSON.parse(fs.readFileSync(path.join(modelsDir, FAST_LOCAL), 'utf8'));
    return validFast(m) ? m : null;
  } catch {
    return null;
  }
}

// Looks up the published fast model. Returns null when none is published yet (or offline).
async function fetchFastManifest(fetchImpl = fetch) {
  try {
    const res = await fetchImpl(FAST_MANIFEST_URL, { redirect: 'follow' });
    if (!res.ok) return null;
    const m = await res.json();
    return validFast(m) ? { id: 'swapperFast', label: 'Fast face swap', pack: 'fast', ...m } : null;
  } catch {
    return null;
  }
}

// The AI voices (trained for this app from computer-made speech) and the two shared speech models
// they need are published the same way, under one manifest:
//   { version, files: [{ file, label, url, sha256, bytes }], voices: { 'young-man': { file, sr, f0 } } }
const VOICE_MANIFEST_URL =
  'https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/download/models-voice/voice-models.json';
const VOICE_LOCAL = 'voice-models.json';
const VOICE_IDS = ['young-man', 'young-woman', 'older-man', 'older-woman'];

function validVoiceFile(m) {
  return (
    m &&
    typeof m.file === 'string' &&
    /^[\w.-]+\.onnx$/.test(m.file) &&
    typeof m.url === 'string' &&
    /^https:\/\/(github\.com|huggingface\.co|objects\.githubusercontent\.com)\//.test(m.url) &&
    /^[0-9a-f]{64}$/.test(m.sha256) &&
    Number.isInteger(m.bytes) &&
    m.bytes > 0 &&
    m.bytes < 5e8
  );
}

function validVoiceManifest(m) {
  if (!m || !Array.isArray(m.files) || !m.files.every(validVoiceFile) || !m.voices || typeof m.voices !== 'object') return null;
  const names = new Set(m.files.map((f) => f.file));
  if (!names.has('contentvec.onnx') || !names.has('rmvpe.onnx')) return null;
  const voices = {};
  for (const id of VOICE_IDS) {
    const v = m.voices[id];
    if (v && names.has(v.file) && Number(v.sr) >= 16000 && Number(v.f0) >= 0) voices[id] = { file: v.file, sr: Number(v.sr), f0: Number(v.f0) };
  }
  if (!Object.keys(voices).length) return null;
  return { version: m.version || 1, files: m.files, voices };
}

function voiceLocal(modelsDir) {
  try {
    return validVoiceManifest(JSON.parse(fs.readFileSync(path.join(modelsDir, VOICE_LOCAL), 'utf8')));
  } catch {
    return null;
  }
}

async function fetchVoiceManifest(fetchImpl = fetch) {
  try {
    const res = await fetchImpl(VOICE_MANIFEST_URL, { redirect: 'follow' });
    if (!res.ok) return null;
    return validVoiceManifest(await res.json());
  } catch {
    return null;
  }
}

function packFiles(pack, extra = []) {
  return [...MODELS, ...extra].filter((m) => m.pack === pack);
}

function packBytes(pack) {
  return packFiles(pack).reduce((sum, m) => sum + m.bytes, 0);
}

// A file counts as present when its size matches; the hash was checked when it was downloaded.
function status(modelsDir) {
  const present = (m) => {
    try {
      return fs.statSync(path.join(modelsDir, m.file)).size === m.bytes;
    } catch {
      return false;
    }
  };
  const fast = fastLocal(modelsDir);
  const voice = voiceLocal(modelsDir);
  const voiceFilesOk = Boolean(voice && voice.files.every(present));
  return {
    core: packFiles('core').every(present),
    boost: packFiles('boost').every(present),
    fast: Boolean(fast && present(fast)),
    fastFile: fast && present(fast) ? fast.file : null,
    voices: voiceFilesOk ? voice.voices : null,
    voicesVersion: voiceFilesOk ? voice.version : null,
    coreBytes: packBytes('core'),
    boostBytes: packBytes('boost'),
  };
}

class Downloader {
  constructor(modelsDir) {
    this.modelsDir = modelsDir;
    this.controller = null;
  }

  get busy() {
    return Boolean(this.controller);
  }

  cancel() {
    if (this.controller) this.controller.abort();
  }

  // Downloads every missing file of a pack. onProgress({ done, total, file }) fires as bytes arrive.
  async download(pack, onProgress, fetchImpl = fetch) {
    if (this.controller) throw new Error('A download is already running');
    let extra = [];
    if (pack === 'fast') {
      const fast = await fetchFastManifest(fetchImpl);
      if (!fast) throw new Error("The fast model isn't published yet. Keep using Best for now.");
      extra = [fast];
    }
    let voicePack = null;
    if (pack === 'voices') {
      voicePack = await fetchVoiceManifest(fetchImpl);
      if (!voicePack) throw new Error("The AI voices aren't published yet. The voice effects work in the meantime.");
      extra = voicePack.files.map((f) => ({ ...f, label: f.label || 'AI voice', pack: 'voices' }));
    }
    const files = packFiles(pack, extra);
    if (!files.length) throw new Error(`Unknown model pack ${pack}`);
    fs.mkdirSync(this.modelsDir, { recursive: true });

    const total = files.reduce((sum, m) => sum + m.bytes, 0);
    let done = 0;
    this.controller = new AbortController();
    try {
      for (const model of files) {
        const target = path.join(this.modelsDir, model.file);
        if (fs.existsSync(target) && fs.statSync(target).size === model.bytes) {
          done += model.bytes;
          onProgress({ done, total, file: model.label });
          continue;
        }
        await this.fetchOne(model, target, (n) => {
          done += n;
          onProgress({ done, total, file: model.label });
        }, fetchImpl);
      }
    } finally {
      this.controller = null;
    }
    if (pack === 'voices') {
      fs.writeFileSync(path.join(this.modelsDir, VOICE_LOCAL), JSON.stringify(voicePack));
    }
    if (pack === 'fast') {
      const { file, url, sha256, bytes, version } = extra[0];
      fs.writeFileSync(path.join(this.modelsDir, FAST_LOCAL), JSON.stringify({ file, url, sha256, bytes, version }));
    }
    return status(this.modelsDir);
  }

  async fetchOne(model, target, onBytes, fetchImpl) {
    const partial = `${target}.part`;
    const res = await fetchImpl(model.url, { signal: this.controller.signal, redirect: 'follow' });
    if (!res.ok || !res.body) throw new Error(`Download of ${model.label} failed (HTTP ${res.status})`);

    const hash = crypto.createHash('sha256');
    const out = fs.createWriteStream(partial);
    let received = 0;
    try {
      for await (const chunk of Readable.fromWeb(res.body)) {
        hash.update(chunk);
        received += chunk.length;
        if (!out.write(chunk)) await new Promise((r) => out.once('drain', r));
        onBytes(chunk.length);
      }
      await new Promise((resolve, reject) => out.end((err) => (err ? reject(err) : resolve())));
    } catch (err) {
      out.destroy();
      fs.rmSync(partial, { force: true });
      if (err.name === 'AbortError') throw new Error('Download cancelled');
      throw err;
    }

    const digest = hash.digest('hex');
    if (received !== model.bytes || digest !== model.sha256) {
      fs.rmSync(partial, { force: true });
      throw new Error(`${model.label} arrived damaged. Please try the download again.`);
    }
    fs.renameSync(partial, target);
  }
}

module.exports = { MODELS, Downloader, status, packBytes, fetchFastManifest, FAST_MANIFEST_URL, fetchVoiceManifest, validVoiceManifest, VOICE_MANIFEST_URL, VOICE_IDS };
