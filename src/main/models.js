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

function packFiles(pack) {
  return MODELS.filter((m) => m.pack === pack);
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
  return {
    core: packFiles('core').every(present),
    boost: packFiles('boost').every(present),
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
    const files = packFiles(pack);
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

module.exports = { MODELS, Downloader, status, packBytes };
