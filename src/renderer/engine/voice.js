// Voice studio: your microphone -> the chosen voice -> the Calls window (and your headphones if you
// want to hear yourself). Each preset has two forms:
//   - a trained AI voice (when the AI voices are downloaded): a separate engine process, ~0.4 s delay
//   - a voice effect (pitch + voice size + age), instant, used until the AI voice is ready
// "My voices" (id 'mine:<id>') are made from your own uploads and are always AI; until the engine is
// ready your own voice plays.
import { VOICES, voiceParams } from './voice-dsp.js';
import { AiVoiceEngine } from './voice-ai.js';

export class VoiceStudio {
  constructor({ onState = () => {}, api = null } = {}) {
    this.onState = onState;
    this.ai = api && api.voiceEngineStart ? new AiVoiceEngine(api, { onInfo: (info) => this.aiInfo(info) }) : null;
    this.aiVoices = null; // installed AI voices: { 'young-man': { file, sr, f0 }, ... }
    this.aiWanted = true;
    this.myVoices = {}; // prepared voices from your uploads: { <id>: { name, seconds, quality } }
    this.myReady = false; // the models for them are installed
    this.aiState = 'off'; // off | loading | on | error
    this.aiError = '';
    this.aiNode = null;
    this.aiGain = null;
    this.fxGain = null;
    this.gpuCount = 0;
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

  // Your microphone as it is (for "use my natural voice"), or null until it's on.
  get rawTrack() {
    return this.stream ? this.stream.getAudioTracks()[0] || null : null;
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
      // Two paths, one heard at a time: the voice effect and the AI voice.
      const bus = this.ctx.createGain();
      this.fxGain = this.ctx.createGain();
      this.aiGain = this.ctx.createGain();
      this.aiGain.gain.value = 0;
      src.connect(this.node);
      this.node.connect(this.fxGain).connect(bus);
      if (this.ai) {
        await this.ctx.audioWorklet.addModule('voice-ai-worklet.js');
        this.aiNode = new AudioWorkletNode(this.ctx, 'brainbox-ai-voice', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        this.aiNode.port.onmessage = (e) => {
          this.aiBuffer = e.data;
        };
        src.connect(this.aiNode);
        this.aiNode.connect(this.aiGain).connect(bus);
        this.givePort();
      }
      bus.connect(this.dest);
      bus.connect(this.analyser);
      bus.connect(this.monitorGain).connect(this.ctx.destination);
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
    this.aiNode = null;
    this.dest = null;
    this.portGiven = null;
    if (this.aiState !== 'error') this.aiState = 'off';
    if (this.ai && this.ai.control) this.ai.reset();
    this.onState();
  }

  // ---------- AI voices ----------

  // Called with the models status: which AI voices are installed, and whether to use them.
  setAiVoices(voices, wanted = true) {
    this.aiVoices = voices || null;
    this.aiWanted = Boolean(wanted);
    this.apply();
  }

  get aiVoice() {
    if (!this.ai) return null;
    if (this.voice.startsWith('mine:')) {
      const id = this.voice.slice(5);
      return this.myReady && this.myVoices[id] ? { kind: 'mine', id } : null;
    }
    const v = this.aiVoices && this.aiVoices[this.voice];
    return this.aiWanted && v ? { kind: 'trained', ...v } : null;
  }

  setMyVoices(voices, ready) {
    this.myVoices = voices || {};
    this.myReady = Boolean(ready);
    if (this.voice.startsWith('mine:') && !this.myVoices[this.voice.slice(5)]) this.voice = 'own';
    this.apply();
  }

  // True while the AI voice is what people hear.
  get aiLive() {
    return this.aiState === 'on' && Boolean(this.aiVoice) && this.running;
  }

  // Hands the engine's audio port to the worklet. A port can only move once, so a new worklet (after
  // the microphone was turned off and on) asks the engine for a fresh pair.
  givePort() {
    if (!this.aiNode || !this.ai) return;
    const port = this.ai.audioPort;
    if (port) {
      this.ai.audioPort = null;
      this.portGiven = port;
      this.aiNode.port.postMessage({ type: 'port', port }, [port]);
    } else if (!this.portGiven && this.ai.control) {
      this.portGiven = 'asked';
      this.ai.api.voiceEngineStart();
    }
  }

  aiInfo(info) {
    if (info.ports) {
      if (this.portGiven === 'asked') this.portGiven = null;
      this.givePort();
    }
    if (info.speakerF0) this.speakerF0 = info.speakerF0;
    if (info.exited) {
      this.aiState = 'off';
      this.route();
    }
    if (info.error) this.aiError = info.error;
    this.onState();
  }

  async useAi() {
    const v = this.aiVoice;
    if (!v || !this.running) return;
    const key = voiceKey(v);
    if (this.aiState === 'error' && this.aiFailed === key) return; // don't retry on its own
    if (this.aiState === 'on' && this.ai.voiceFile === key) return;
    const token = (this.aiToken = (this.aiToken || 0) + 1);
    if (!this.ai.ready || this.ai.voiceFile !== key) {
      this.aiState = 'loading';
      this.onState();
    }
    try {
      await this.ai.start({ gpuCount: this.gpuCount });
      this.givePort();
      await this.ai.setVoice(v);
      if (token !== this.aiToken) return;
      this.aiState = 'on';
      this.aiFailed = null;
      this.aiError = '';
    } catch (err) {
      if (token !== this.aiToken) return;
      this.aiState = 'error';
      this.aiFailed = key;
      this.aiError = (err && err.message) || String(err);
    }
    this.route();
    this.onState();
  }

  // Which path people hear. The effect keeps playing until the AI voice is ready.
  route() {
    if (!this.ctx) return;
    const ai = this.aiLive;
    const t = this.ctx.currentTime;
    this.fxGain.gain.setTargetAtTime(ai ? 0 : 1, t, 0.03);
    this.aiGain.gain.setTargetAtTime(ai ? 1 : 0, t, 0.03);
    if (this.aiNode) this.aiNode.port.postMessage({ type: 'active', on: ai });
    if (!ai && this.ai && this.ai.control) this.ai.reset();
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

  // Try the AI voice again after it failed (the user asked).
  retryAi() {
    this.aiFailed = null;
    this.aiState = 'off';
    this.apply();
  }

  setVoice(id) {
    this.voice = VOICES[id] || (typeof id === 'string' && id.startsWith('mine:') && this.myVoices[id.slice(5)]) ? id : 'own';
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
    if (this.aiVoice && this.running) this.useAi();
    else if (this.aiState !== 'off') {
      this.aiState = this.aiState === 'error' ? 'error' : 'off';
      this.route();
    }
  }

  // What the voice panel shows about the AI voice.
  aiSummary() {
    if (!this.ai) return '';
    if (!this.aiVoices) return 'none';
    if (!this.aiWanted) return 'disabled';
    if (!this.aiVoice) return this.voice === 'own' ? 'idle' : this.voice.startsWith('mine:') ? 'mine-missing' : 'missing';
    return this.aiState;
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

export function voiceKey(v) {
  return v.kind === 'mine' ? `mine:${v.id}` : `trained:${v.file}`;
}

export { VOICES };
