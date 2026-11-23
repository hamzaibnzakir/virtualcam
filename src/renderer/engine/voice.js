// Voice studio: your microphone -> the chosen voice -> the Calls window (and your headphones if you
// want to hear yourself). Preset voices are voice effects for now (pitch + voice size + age); trained
// AI voices replace them when they're installed.
import { VOICES, voiceParams } from './voice-dsp.js';

export class VoiceStudio {
  constructor({ onState = () => {} } = {}) {
    this.onState = onState;
    this.voice = 'own';
    this.micId = '';
    this.monitor = false;
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.dest = null;
    this.monitorGain = null;
    this.pitches = [];
    this.speakerF0 = 0;
    this.level = 0;
    this.error = '';
    this.silent = null;
  }

  get running() {
    return Boolean(this.ctx && this.stream);
  }

  // The track the calls receive: the changed voice, or silence until the microphone is on.
  get track() {
    if (this.dest) return this.dest.stream.getAudioTracks()[0];
    if (!this.silent) {
      const ctx = new AudioContext();
      const d = ctx.createMediaStreamDestination();
      this.silent = d.stream.getAudioTracks()[0];
    }
    return this.silent;
  }

  async start() {
    if (this.running) return this.track;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          deviceId: this.micId ? { exact: this.micId } : undefined,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      });
      this.ctx = new AudioContext({ latencyHint: 'interactive', sampleRate: 48000 });
      await this.ctx.audioWorklet.addModule('voice-worklet.js');
      const src = this.ctx.createMediaStreamSource(this.stream);
      this.node = new AudioWorkletNode(this.ctx, 'brainbox-voice', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
      this.node.port.onmessage = (e) => this.collectPitch(e.data.pitches || []);
      this.dest = this.ctx.createMediaStreamDestination();
      this.monitorGain = this.ctx.createGain();
      this.monitorGain.gain.value = this.monitor ? 1 : 0;
      this.analyser = this.ctx.createAnalyser();
      this.analyser.fftSize = 512;
      src.connect(this.node);
      this.node.connect(this.dest);
      this.node.connect(this.analyser);
      this.node.connect(this.monitorGain).connect(this.ctx.destination);
      this.error = '';
      this.apply();
    } catch (err) {
      this.error = (err && err.message) || String(err);
      this.stop();
    }
    this.onState();
    return this.track;
  }

  stop() {
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close().catch(() => {});
    this.stream = null;
    this.ctx = null;
    this.node = null;
    this.dest = null;
    this.onState();
  }

  collectPitch(list) {
    // A running median of your speaking pitch, so presets aim at the same target for anyone.
    this.pitches.push(...list.filter((f) => f > 65 && f < 420));
    if (this.pitches.length > 400) this.pitches.splice(0, this.pitches.length - 400);
    if (this.pitches.length >= 20) {
      const sorted = [...this.pitches].sort((a, b) => a - b);
      const median = sorted[Math.floor(sorted.length / 2)];
      if (!this.speakerF0 || Math.abs(median - this.speakerF0) / this.speakerF0 > 0.05) {
        this.speakerF0 = median;
        this.apply();
        this.onState();
      }
    }
  }

  setVoice(id) {
    this.voice = VOICES[id] ? id : 'own';
    this.apply();
  }

  setMonitor(on) {
    this.monitor = Boolean(on);
    if (this.monitorGain) this.monitorGain.gain.value = this.monitor ? 1 : 0;
  }

  async setMic(id) {
    this.micId = id || '';
    if (this.running) {
      this.stop();
      await this.start();
    }
  }

  apply() {
    if (this.node) this.node.port.postMessage(voiceParams(this.voice, this.speakerF0));
  }

  // 0..1 loudness of the changed voice, for a level meter.
  meter() {
    if (!this.analyser) return 0;
    const data = new Float32Array(this.analyser.fftSize);
    this.analyser.getFloatTimeDomainData(data);
    let s = 0;
    for (const v of data) s += v * v;
    return Math.min(1, Math.sqrt(s / data.length) * 4);
  }
}

export { VOICES };
