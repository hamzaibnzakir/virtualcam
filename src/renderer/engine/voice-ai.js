// The studio's side of the AI voice engine (a separate process, see src/main/voice/worker.js).
// Control messages go over one port; the other port is handed to the audio worklet, which streams the
// microphone to the engine and plays what comes back.
export class AiVoiceEngine {
  constructor(api, { onInfo = () => {} } = {}) {
    this.api = api;
    this.onInfo = onInfo;
    this.control = null;
    this.audioPort = null;
    this.loaded = null; // { device, ms }
    this.voiceFile = null;
    this.waiters = [];
    this.starting = null;
    this.error = '';
    window.addEventListener('message', (e) => {
      if (e.source !== window || !e.data || e.data.type !== 'brainbox-voice-ports' || e.ports.length < 2) return;
      this.attach(e.ports[0], e.ports[1]);
    });
    if (api.onVoiceEngineExit) {
      api.onVoiceEngineExit(() => {
        this.control = null;
        this.audioPort = null;
        this.loaded = null;
        this.voiceFile = null;
        this.starting = null;
        this.error = 'The voice engine stopped. It restarts when you pick an AI voice again.';
        this.onInfo({ exited: true });
      });
    }
  }

  get ready() {
    return Boolean(this.loaded && this.voiceFile);
  }

  attach(control, audio) {
    if (this.control) this.control.close();
    this.control = control;
    this.audioPort = audio;
    control.onmessage = (e) => this.receive(e.data);
    control.start();
    this.onInfo({ ports: true });
  }

  receive(msg) {
    if (!msg) return;
    if (msg.type === 'info') this.onInfo(msg);
    const left = [];
    for (const w of this.waiters) {
      if (msg.type === w.type || (msg.type === 'error' && msg.request === w.request)) {
        clearTimeout(w.timer);
        if (msg.type === 'error') w.reject(new Error(msg.message));
        else w.resolve(msg);
      } else left.push(w);
    }
    this.waiters = left;
  }

  request(message, type, ms = 120000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.waiters = this.waiters.filter((w) => w.timer !== timer);
        reject(new Error('The voice engine did not answer in time'));
      }, ms);
      this.waiters.push({ type, request: message.type, resolve, reject, timer });
      this.control.postMessage(message);
    });
  }

  // Starts the engine process and loads the shared models (once).
  async start({ gpuCount = 0 } = {}) {
    if (this.loaded && this.control) return this.loaded;
    if (!this.starting) {
      this.starting = (async () => {
        const gotPorts = new Promise((resolve) => {
          const prev = this.onInfo;
          this.onInfo = (info) => {
            prev(info);
            if (info.ports) {
              this.onInfo = prev;
              resolve();
            }
          };
        });
        const res = await this.api.voiceEngineStart();
        if (!res || !res.ok) throw new Error('The voice engine could not start');
        await gotPorts;
        const loaded = await this.request({ type: 'load', modelsDir: res.modelsDir, preferred: 'auto', gpuCount }, 'loaded', 180000);
        this.loaded = { device: loaded.device, ms: loaded.ms };
        this.voiceFile = null;
        this.error = '';
        return this.loaded;
      })();
      this.starting.catch((err) => {
        this.error = err.message;
        this.starting = null;
      });
    }
    return this.starting;
  }

  async setVoice(voice) {
    await this.start();
    if (this.voiceFile === voice.file) return;
    await this.request({ type: 'voice', voice }, 'voice', 120000);
    this.voiceFile = voice.file;
  }

  async stats() {
    if (!this.control) return null;
    return this.request({ type: 'stats' }, 'stats', 5000).catch(() => null);
  }

  reset() {
    if (this.control) this.control.postMessage({ type: 'reset' });
  }
}
