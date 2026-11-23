// Voice changer DSP: pitch shift with independent formant (voice-size) shift, plus age effects.
// Phase-vocoder pitch shifting (after Bernsee); the spectral envelope (cepstrum) is taken off before
// the shift and put back after, scaled by the formant ratio, so a voice can go higher without sounding
// like a chipmunk, or a man's voice can take on a woman's voice size and pitch together.
// Also estimates the speaker's own pitch (autocorrelation from the same FFT), so presets can aim at a
// target pitch whoever is speaking. Plain JS, used inside an AudioWorklet and in tests.

const TWO_PI = Math.PI * 2;

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
      this.cos[i] = Math.cos((TWO_PI * i) / n);
      this.sin[i] = Math.sin((TWO_PI * i) / n);
    }
  }

  // In place. inverse=true computes the unscaled inverse transform.
  run(re, im, inverse = false) {
    const { n, rev } = this;
    for (let i = 0; i < n; i += 1) {
      const j = rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k += 1) {
          const wr = this.cos[k * step];
          const wi = sign * this.sin[k * step];
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

export class VoiceDSP {
  constructor(sampleRate, { size = 2048, overlap = 4 } = {}) {
    this.sr = sampleRate;
    this.N = size;
    this.osamp = overlap;
    this.hop = size / overlap;
    this.half = size / 2;
    this.fft = new FFT(size);
    const N = size;
    this.window = new Float64Array(N);
    for (let i = 0; i < N; i += 1) this.window[i] = 0.5 - 0.5 * Math.cos((TWO_PI * i) / N);
    this.inFifo = new Float64Array(N);
    this.outFifo = new Float64Array(N);
    this.accum = new Float64Array(2 * N);
    this.lastPhase = new Float64Array(this.half + 1);
    this.sumPhase = new Float64Array(this.half + 1);
    this.re = new Float64Array(N);
    this.im = new Float64Array(N);
    this.cre = new Float64Array(N);
    this.cim = new Float64Array(N);
    this.mag = new Float64Array(this.half + 1);
    this.freq = new Float64Array(this.half + 1);
    this.env = new Float64Array(this.half + 1);
    this.synMag = new Float64Array(this.half + 1);
    this.synFreq = new Float64Array(this.half + 1);
    this.rover = N - this.hop;
    this.latency = N - this.hop;
    this.frames = 0;
    this.time = 0;
    this.jitter = 0;
    this.params = { pitch: 1, formant: 1, vibrato: 0, vibratoRate: 5.5, jitter: 0, breath: 0, tone: 0, bypass: true };
    this.onPitch = null; // (hz | 0) per frame: the speaker's own pitch (0 = no voice)
    this.lifter = Math.max(8, Math.round(sampleRate / 1000)); // keep ~1 ms of cepstrum: the formant envelope
  }

  set(params) {
    Object.assign(this.params, params);
  }

  // Processes `input` into `output` (same length, any length). Delay: this.latency samples.
  process(input, output) {
    const { N, hop } = this;
    for (let i = 0; i < input.length; i += 1) {
      this.inFifo[this.rover] = input[i];
      output[i] = this.outFifo[this.rover - (N - hop)];
      this.rover += 1;
      if (this.rover >= N) {
        this.rover = N - hop;
        this.frame();
      }
    }
  }

  frame() {
    const { N, hop, half, osamp, re, im, window: win, sr } = this;
    const p = this.params;
    const freqPerBin = sr / N;
    const expct = (TWO_PI * hop) / N;
    for (let k = 0; k < N; k += 1) {
      re[k] = this.inFifo[k] * win[k];
      im[k] = 0;
    }
    this.fft.run(re, im);

    // Analysis: magnitude and true frequency of each bin.
    for (let k = 0; k <= half; k += 1) {
      const mag = Math.hypot(re[k], im[k]);
      const phase = Math.atan2(im[k], re[k]);
      let d = phase - this.lastPhase[k];
      this.lastPhase[k] = phase;
      d -= k * expct;
      d -= TWO_PI * Math.round(d / TWO_PI);
      this.mag[k] = mag;
      this.freq[k] = k * freqPerBin + ((osamp * d) / TWO_PI) * freqPerBin;
    }
    this.estimatePitch();

    if (p.bypass) {
      this.resynthesise(this.mag, this.freq, 1);
    } else {
      this.envelope();
      // Age: slow wobble (vibrato) plus a little random unsteadiness (jitter).
      this.time += hop / sr;
      this.jitter = this.jitter * 0.9 + (Math.random() * 2 - 1) * 0.1;
      const wobble = p.vibrato * Math.sin(TWO_PI * p.vibratoRate * this.time) + p.jitter * this.jitter;
      const ratio = p.pitch * 2 ** (wobble / 12);
      const { synMag, synFreq, env, mag, freq } = this;
      synMag.fill(0);
      synFreq.fill(0);
      for (let k = 0; k <= half; k += 1) {
        const target = Math.round(k * ratio);
        if (target > half) break;
        synMag[target] += mag[k] / env[k];
        synFreq[target] = freq[k] * ratio;
      }
      // Put the envelope back, scaled by the formant ratio, with an optional darker/brighter tilt.
      const avg = this.excitationLevel(synMag);
      for (let k = 0; k <= half; k += 1) {
        const src = k / p.formant;
        const i0 = Math.floor(src);
        const e = i0 >= half ? env[half] * 1e-3 : env[i0] + (env[Math.min(half, i0 + 1)] - env[i0]) * (src - i0);
        const hz = k * freqPerBin;
        const tilt = p.tone ? (1 + (hz / 3000) ** 2) ** (p.tone / 2) : 1;
        synMag[k] *= e * tilt;
        // Breath: a little noise shaped like the voice, above ~800 Hz.
        if (p.breath && hz > 800) synMag[k] += p.breath * avg * e * tilt * Math.random();
        if (p.breath && hz > 800 && synFreq[k] === 0) synFreq[k] = hz + (Math.random() - 0.5) * freqPerBin;
      }
      this.resynthesise(synMag, synFreq, 1);
    }
  }

  excitationLevel(synMag) {
    let s = 0;
    for (let k = 1; k < synMag.length; k += 1) s += synMag[k];
    return s / synMag.length;
  }

  // Smooth spectral envelope (formants) from the cepstrum.
  envelope() {
    const { N, half, cre, cim, mag, env, lifter } = this;
    for (let k = 0; k <= half; k += 1) {
      cre[k] = Math.log(mag[k] + 1e-9);
      cim[k] = 0;
      if (k > 0 && k < half) {
        cre[N - k] = cre[k];
        cim[N - k] = 0;
      }
    }
    this.fft.run(cre, cim, true);
    for (let q = 0; q < N; q += 1) {
      const keep = q < lifter || q > N - lifter ? 1 : q === lifter || q === N - lifter ? 0.5 : 0;
      cre[q] = (cre[q] / N) * keep;
      cim[q] = 0;
    }
    this.fft.run(cre, cim);
    for (let k = 0; k <= half; k += 1) env[k] = Math.max(1e-9, Math.exp(cre[k]));
  }

  resynthesise(synMag, synFreq) {
    const { N, half, hop, osamp, re, im, window: win, sr } = this;
    const freqPerBin = sr / N;
    const expct = (TWO_PI * hop) / N;
    for (let k = 0; k <= half; k += 1) {
      let d = synFreq[k] - k * freqPerBin;
      d /= freqPerBin;
      d = (TWO_PI * d) / osamp;
      d += k * expct;
      this.sumPhase[k] += d;
      const ph = this.sumPhase[k];
      re[k] = synMag[k] * Math.cos(ph);
      im[k] = synMag[k] * Math.sin(ph);
    }
    for (let k = half + 1; k < N; k += 1) {
      re[k] = re[N - k];
      im[k] = -im[N - k];
    }
    this.fft.run(re, im, true);
    for (let k = 0; k < N; k += 1) this.accum[k] += (2 * win[k] * re[k]) / (half * osamp);
    for (let k = 0; k < hop; k += 1) this.outFifo[k] = this.accum[k];
    this.accum.copyWithin(0, hop, hop + 2 * N - hop);
    this.accum.fill(0, 2 * N - hop);
    this.inFifo.copyWithin(0, hop);
  }

  // The speaker's own pitch from this frame's autocorrelation (power spectrum -> inverse FFT).
  estimatePitch() {
    if (!this.onPitch) return;
    const { N, half, cre, cim, mag, sr } = this;
    let energy = 0;
    for (let k = 0; k <= half; k += 1) {
      const pw = mag[k] * mag[k];
      energy += pw;
      cre[k] = pw;
      cim[k] = 0;
      if (k > 0 && k < half) {
        cre[N - k] = pw;
        cim[N - k] = 0;
      }
    }
    if (energy < 1e-4) {
      this.onPitch(0);
      return;
    }
    this.fft.run(cre, cim, true);
    const r0 = cre[0];
    const lo = Math.floor(sr / 420);
    const hi = Math.min(Math.ceil(sr / 65), N / 2 - 1);
    let best = 0;
    let bestLag = 0;
    for (let lag = lo; lag <= hi; lag += 1) {
      // Hann window shortens long lags; correct for it roughly.
      const v = cre[lag] / r0 / (1 - lag / N);
      if (v > best) {
        best = v;
        bestLag = lag;
      }
    }
    if (best < 0.45 || !bestLag) {
      this.onPitch(0);
      return;
    }
    // Parabolic interpolation around the peak.
    const a = cre[bestLag - 1];
    const b = cre[bestLag];
    const c = cre[bestLag + 1];
    const shift = (a - c) / (2 * (a - 2 * b + c) || 1);
    this.onPitch(sr / (bestLag + (Number.isFinite(shift) ? shift : 0)));
  }
}

// The four preset voices: target pitch (Hz), voice size relative to a man's (formant scale),
// and age effects. The pitch shift is worked out from the speaker's own measured pitch.
export const VOICES = {
  own: { label: 'My own voice', bypass: true },
  'young-man': { label: 'Young man', f0: 120, size: 1.0 },
  'young-woman': { label: 'Young woman', f0: 215, size: 1.17 },
  'older-man': { label: 'Older man', f0: 105, size: 0.97, vibrato: 0.22, jitter: 0.5, breath: 0.12, tone: -0.35 },
  'older-woman': { label: 'Older woman', f0: 190, size: 1.11, vibrato: 0.28, jitter: 0.5, breath: 0.12, tone: -0.25 },
};

// DSP settings for a voice, given the speaker's measured median pitch.
export function voiceParams(id, speakerF0) {
  const v = VOICES[id] || VOICES.own;
  if (v.bypass) return { bypass: true };
  const f0 = speakerF0 > 60 && speakerF0 < 400 ? speakerF0 : 130;
  const speakerSize = f0 > 165 ? 1.17 : 1.0; // a higher speaking pitch usually means a smaller voice tract
  return {
    bypass: false,
    pitch: Math.min(2.2, Math.max(0.45, v.f0 / f0)),
    formant: v.size / speakerSize,
    vibrato: v.vibrato || 0,
    vibratoRate: 5.5,
    jitter: v.jitter || 0,
    breath: v.breath || 0,
    tone: v.tone || 0,
  };
}
