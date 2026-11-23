'use strict';

// Signal helpers for the AI voice engine: a streaming resampler, the log-mel spectrogram the pitch
// model (RMVPE) expects, and the pitch decoding it needs. These match RVC's own Python code, so voices
// trained with RVC/Applio sound the same here.

// ---------- resampling ----------

function besselI0(x) {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 50; k += 1) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

// Streaming rational resampler (up by `up`, down by `down`) with a Kaiser-windowed sinc filter.
// Push any number of samples at a time; it returns the output samples that are ready.
class Resampler {
  constructor(from, to, { taps = 128, beta = 8.6 } = {}) {
    const g = gcd(from, to);
    this.up = to / g;
    this.down = from / g;
    this.taps = taps; // filter taps per phase
    const cutoff = 0.5 * 0.94 * Math.min(1 / this.up, 1 / this.down); // in cycles per upsampled sample
    const len = taps * this.up;
    const mid = (len - 1) / 2;
    const h = new Float32Array(len);
    const norm = besselI0(beta);
    for (let i = 0; i < len; i += 1) {
      const t = i - mid;
      const sinc = t === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * t) / (Math.PI * t);
      const r = (2 * i) / (len - 1) - 1;
      h[i] = sinc * (besselI0(beta * Math.sqrt(Math.max(0, 1 - r * r))) / norm) * this.up;
    }
    // Phase p uses taps h[p], h[p + up], ... applied to x[i], x[i - 1], ...
    this.phases = [];
    for (let p = 0; p < this.up; p += 1) {
      const ph = new Float32Array(taps);
      for (let k = 0; k < taps; k += 1) ph[k] = h[p + k * this.up];
      this.phases.push(ph);
    }
    this.delay = mid / this.up; // in input samples (filter's group delay)
    this.history = new Float32Array(taps - 1);
    this.pos = 0; // next output position, in upsampled samples, relative to the history start
  }

  push(input) {
    const taps = this.taps;
    const buf = new Float32Array(this.history.length + input.length);
    buf.set(this.history);
    buf.set(input, this.history.length);
    const out = [];
    // x index of the newest sample a filter step may use: buf.length - 1
    for (;;) {
      const base = Math.floor(this.pos / this.up);
      const i = base + taps - 1; // newest input used
      if (i >= buf.length) break;
      const ph = this.phases[this.pos % this.up];
      let acc = 0;
      for (let k = 0; k < taps; k += 1) acc += ph[k] * buf[i - k];
      out.push(acc);
      this.pos += this.down;
    }
    // Keep the last taps-1 samples; re-base the position on them.
    const consumed = buf.length - (taps - 1);
    this.history = buf.slice(consumed);
    this.pos -= consumed * this.up;
    return Float32Array.from(out);
  }
}

function gcd(a, b) {
  return b ? gcd(b, a % b) : a;
}

// ---------- FFT ----------

class FFT {
  constructor(n) {
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i += 1) {
      let r = 0;
      for (let b = 0; b < bits; b += 1) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i += 1) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
  }

  // In-place forward transform of (re, im).
  run(re, im) {
    const { n, rev } = this;
    for (let i = 0; i < n; i += 1) {
      const j = rev[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let size = 2; size <= n; size *= 2) {
      const half = size / 2;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k += 1) {
          const wr = this.cos[k * step];
          const wi = this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}

// ---------- RMVPE input: log-mel spectrogram ----------
// RVC: 16 kHz, n_fft = win = 1024, hop 160, periodic Hann, centred with reflect padding, magnitude,
// librosa mel filters (htk mel scale, Slaney area normalisation, 30-8000 Hz, 128 bands), log(max(x, 1e-5)).

const hzToMelHtk = (f) => 2595 * Math.log10(1 + f / 700);
const melToHzHtk = (m) => 700 * (10 ** (m / 2595) - 1);

function melFilters({ sr = 16000, nFft = 1024, nMels = 128, fmin = 30, fmax = 8000 } = {}) {
  const bins = nFft / 2 + 1;
  const fftFreqs = Array.from({ length: bins }, (_, i) => (i * sr) / nFft);
  const lo = hzToMelHtk(fmin);
  const hi = hzToMelHtk(fmax);
  const melF = Array.from({ length: nMels + 2 }, (_, i) => melToHzHtk(lo + ((hi - lo) * i) / (nMels + 1)));
  const weights = new Float32Array(nMels * bins);
  for (let m = 0; m < nMels; m += 1) {
    const d0 = melF[m + 1] - melF[m];
    const d1 = melF[m + 2] - melF[m + 1];
    const enorm = 2 / (melF[m + 2] - melF[m]);
    for (let k = 0; k < bins; k += 1) {
      const lower = (fftFreqs[k] - melF[m]) / d0;
      const upper = (melF[m + 2] - fftFreqs[k]) / d1;
      weights[m * bins + k] = Math.max(0, Math.min(lower, upper)) * enorm;
    }
  }
  return { weights, bins, nMels };
}

class MelSpectrogram {
  constructor({ nFft = 1024, hop = 160, nMels = 128 } = {}) {
    this.nFft = nFft;
    this.hop = hop;
    this.fft = new FFT(nFft);
    this.window = new Float64Array(nFft);
    for (let i = 0; i < nFft; i += 1) this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nFft);
    this.mel = melFilters({ nFft, nMels });
    this.re = new Float64Array(nFft);
    this.im = new Float64Array(nFft);
    this.mag = new Float64Array(nFft / 2 + 1);
  }

  // audio (16 kHz) -> Float32Array [nMels, frames] (row-major) with frames = floor(len / hop) + 1.
  compute(audio) {
    const { nFft, hop } = this;
    const pad = nFft / 2;
    const n = audio.length;
    const frames = Math.floor(n / hop) + 1;
    const at = (i) => {
      // reflect padding, as torch.stft(center=True)
      let j = i - pad;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * (n - 1) - j;
      return audio[Math.max(0, Math.min(n - 1, j))];
    };
    const { weights, bins, nMels } = this.mel;
    const out = new Float32Array(nMels * frames);
    for (let f = 0; f < frames; f += 1) {
      const start = f * hop;
      for (let i = 0; i < nFft; i += 1) {
        this.re[i] = at(start + i) * this.window[i];
        this.im[i] = 0;
      }
      this.fft.run(this.re, this.im);
      for (let k = 0; k < bins; k += 1) this.mag[k] = Math.hypot(this.re[k], this.im[k]);
      for (let m = 0; m < nMels; m += 1) {
        let s = 0;
        const row = m * bins;
        for (let k = 0; k < bins; k += 1) s += weights[row + k] * this.mag[k];
        out[m * frames + f] = Math.log(Math.max(s, 1e-5));
      }
    }
    return { data: out, frames };
  }
}

// ---------- RMVPE output: salience -> pitch in Hz ----------

const CENTS = Float32Array.from({ length: 360 }, (_, i) => 20 * i + 1997.3794084376191);

// salience: Float32Array [frames, 360]. Returns f0 in Hz per frame (0 = unvoiced).
function decodePitch(salience, frames, threshold = 0.03) {
  const f0 = new Float32Array(frames);
  for (let t = 0; t < frames; t += 1) {
    const row = t * 360;
    let best = 0;
    let max = -Infinity;
    for (let i = 0; i < 360; i += 1) {
      if (salience[row + i] > max) {
        max = salience[row + i];
        best = i;
      }
    }
    if (max <= threshold) continue;
    let num = 0;
    let den = 0;
    for (let i = Math.max(0, best - 4); i <= Math.min(359, best + 4); i += 1) {
      num += salience[row + i] * CENTS[i];
      den += salience[row + i];
    }
    const cents = num / den;
    f0[t] = 10 * 2 ** (cents / 1200);
  }
  return f0;
}

// RVC's pitch input: coarse = 1..255 on the mel scale between 50 and 1100 Hz (1 = unvoiced).
const F0_MEL_MIN = 1127 * Math.log(1 + 50 / 700);
const F0_MEL_MAX = 1127 * Math.log(1 + 1100 / 700);
function coarsePitch(f0) {
  const out = new BigInt64Array(f0.length);
  for (let i = 0; i < f0.length; i += 1) {
    let mel = 1127 * Math.log(1 + f0[i] / 700);
    if (mel > 0) mel = ((mel - F0_MEL_MIN) * 254) / (F0_MEL_MAX - F0_MEL_MIN) + 1;
    if (mel <= 1) mel = 1;
    if (mel > 255) mel = 255;
    out[i] = BigInt(Math.round(mel));
  }
  return out;
}

function median(values) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
}

function rms(a, from = 0, to = a.length) {
  let s = 0;
  for (let i = from; i < to; i += 1) s += a[i] * a[i];
  return Math.sqrt(s / Math.max(1, to - from));
}

module.exports = { Resampler, FFT, MelSpectrogram, melFilters, decodePitch, coarsePitch, median, rms };

// Joins consecutive blocks of converted audio without clicks: finds where the new block best continues
// the previous one's tail (SOLA) and crossfades over `cross` samples.
class Joiner {
  constructor(cross, search) {
    this.cross = cross;
    this.search = search;
    this.tail = null;
  }

  reset() {
    this.tail = null;
  }

  // Quiet block: fade the previous sound out, then silence.
  silence(n) {
    const out = new Float32Array(n);
    if (this.tail) for (let i = 0; i < Math.min(n, this.cross); i += 1) out[i] = this.tail[i] * (1 - i / this.cross);
    this.tail = null;
    return out;
  }

  // audio must hold at least search + n + cross samples. Returns n samples.
  join(audio, n, gain = 1) {
    const { cross, search } = this;
    if (audio.length < search + n + cross) return null;
    let offset = 0;
    if (this.tail) {
      let best = -Infinity;
      for (let o = 0; o <= search; o += 1) {
        let num = 0;
        let den = 1e-8;
        for (let i = 0; i < cross; i += 1) {
          num += audio[o + i] * this.tail[i];
          den += audio[o + i] * audio[o + i];
        }
        const score = num / Math.sqrt(den);
        if (score > best) {
          best = score;
          offset = o;
        }
      }
    }
    const out = new Float32Array(n);
    for (let i = 0; i < n; i += 1) out[i] = audio[offset + i] * gain;
    for (let i = 0; i < cross; i += 1) {
      const w = Math.sin((0.5 * Math.PI * i) / cross) ** 2;
      out[i] = this.tail ? out[i] * w + this.tail[i] * (1 - w) : out[i] * w;
    }
    this.tail = Float32Array.from(audio.subarray(offset + n, offset + n + cross), (v) => v * gain);
    for (let i = 0; i < n; i += 1) out[i] = Math.max(-1, Math.min(1, out[i]));
    return out;
  }
}

module.exports.Joiner = Joiner;
