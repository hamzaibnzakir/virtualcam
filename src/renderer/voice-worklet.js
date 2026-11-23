// AudioWorklet: the live voice changer (runs on the audio thread). See engine/voice-dsp.js.
import { VoiceDSP } from './engine/voice-dsp.js';

class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.dsp = new VoiceDSP(sampleRate);
    this.pitches = [];
    this.dsp.onPitch = (hz) => {
      if (hz) this.pitches.push(hz);
    };
    this.scratch = new Float64Array(128);
    this.inBuf = new Float64Array(128);
    this.levelIn = 1e-4;
    this.levelOut = 1e-4;
    this.gain = 1;
    this.port.onmessage = (e) => this.dsp.set(e.data);
    this.sent = 0;
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0];
    if (!input || !output || !output[0]) return true;
    const n = input.length;
    if (this.inBuf.length !== n) {
      this.inBuf = new Float64Array(n);
      this.scratch = new Float64Array(n);
    }
    this.inBuf.set(input);
    this.dsp.process(this.inBuf, this.scratch);
    const bypass = this.dsp.params.bypass;
    let si = 0;
    let so = 0;
    for (let i = 0; i < n; i += 1) {
      si += input[i] * input[i];
      so += this.scratch[i] * this.scratch[i];
    }
    // Keep the changed voice as loud as the speaker (pitch and size changes alter the level).
    this.levelIn = this.levelIn * 0.995 + (si / n) * 0.005;
    this.levelOut = this.levelOut * 0.995 + (so / n) * 0.005;
    const want = Math.min(3, Math.max(0.4, Math.sqrt(this.levelIn / Math.max(1e-9, this.levelOut))));
    this.gain += (want - this.gain) * 0.01;
    for (const ch of output) {
      for (let i = 0; i < n; i += 1) ch[i] = bypass ? input[i] : Math.max(-1, Math.min(1, this.scratch[i] * this.gain));
    }
    // Report the speaker's pitches a few times a second.
    this.sent += n;
    if (this.sent > sampleRate / 4) {
      this.sent = 0;
      if (this.pitches.length) this.port.postMessage({ pitches: this.pitches });
      this.pitches = [];
    }
    return true;
  }
}

registerProcessor('brainbox-voice', VoiceProcessor);
