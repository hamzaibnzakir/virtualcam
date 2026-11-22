// Decart realtime adapter: streams the scene in over WebRTC and gets the transformed video back.
// Every AI provider implements the same shape: setTarget(target), stop(), plus state callbacks.
import { createDecartClient, models, resolveFpsNumber } from '@decartai/sdk';

// The SDK logs connection problems itself; the studio already shows them to the user,
// so keep its messages at debug level (visible in DevTools when needed).
const sdkLogger = {
  debug: (message, data) => console.debug('[Decart]', message, data || ''),
  info: (message, data) => console.debug('[Decart]', message, data || ''),
  warn: (message, data) => console.debug('[Decart]', message, data || ''),
  error: (message, data) => console.debug('[Decart]', message, data || ''),
};

const MODEL_IDS = {
  restyle: 'lucy-restyle-latest', // restyle the whole picture from a prompt
  edit: 'lucy-latest', // edit the scene; accepts a reference image
};

export class DecartAI {
  constructor({ getApiKey, getInputStream, onRemoteStream, onState, onError }) {
    this.getApiKey = getApiKey;
    this.getInputStream = getInputStream;
    this.onRemoteStream = onRemoteStream;
    this.onState = onState;
    this.onError = onError;
    this.client = null;
    this.session = null; // { mode, prompt, enhance, refKey }
    this.target = null;
    this.running = false;
    this.dirty = false;
    this.state = 'off';
  }

  static modelFor(mode) {
    return models.realtime(MODEL_IDS[mode] || MODEL_IDS.restyle);
  }

  static inputSizeFor(mode) {
    const model = DecartAI.modelFor(mode);
    return { width: model.width, height: model.height, fps: resolveFpsNumber(model.fps, 30) };
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState(state);
  }

  // target: null (AI off) or { mode, prompt, enhance, refKey, refBlob }
  setTarget(target) {
    this.target = target;
    this.dirty = true;
    this.pump();
  }

  stop() {
    this.setTarget(null);
  }

  async pump() {
    if (this.running) return;
    this.running = true;
    try {
      while (this.dirty) {
        this.dirty = false;
        await this.apply(this.target);
      }
    } finally {
      this.running = false;
    }
  }

  async apply(target) {
    if (!target) {
      this.disconnect();
      this.setState('off');
      return;
    }

    const prompt = (target.prompt || '').trim();
    if (!prompt && !target.refBlob) {
      this.disconnect();
      this.setState('needs-prompt');
      return;
    }

    const live = this.client && this.session && this.client.isConnected();
    if (live && this.session.mode === target.mode) {
      try {
        if (prompt !== this.session.prompt || target.enhance !== this.session.enhance) {
          await this.client.set({ prompt, enhance: target.enhance });
          this.session.prompt = prompt;
          this.session.enhance = target.enhance;
        }
        if (target.mode === 'edit' && target.refKey !== this.session.refKey) {
          await this.client.setImage(target.refBlob || null);
          this.session.refKey = target.refKey;
        }
        return;
      } catch (err) {
        this.onError(`AI update failed: ${messageOf(err)}`);
        // Fall through and reconnect cleanly.
      }
    }

    await this.connect(target, prompt);
  }

  async connect(target, prompt) {
    this.disconnect();

    const apiKey = await this.getApiKey();
    if (!apiKey) {
      this.setState('needs-key');
      return;
    }

    this.setState('connecting');
    const model = DecartAI.modelFor(target.mode);
    const initialState = {};
    if (prompt) initialState.prompt = { text: prompt, enhance: target.enhance };
    // Restyle takes a prompt OR a reference image; edit takes both.
    if (target.refBlob && (target.mode === 'edit' || !prompt)) initialState.image = target.refBlob;

    let rt = null;
    this.connecting = true;
    try {
      const client = createDecartClient({ apiKey, telemetry: false, logger: sdkLogger });
      const stream = this.getInputStream(target.mode);
      rt = await client.realtime.connect(stream, {
        model,
        mirror: false,
        initialState,
        onRemoteStream: (remote) => {
          if (rt === null || this.client === rt) this.onRemoteStream(remote);
        },
        onConnectionChange: (state) => this.handleConnection(rt, state),
      });
      this.connecting = false;

      // AI was switched off or to another mode while connecting: drop this session.
      // (Prompt or image changes are applied by the next pump pass.)
      if (!this.target || this.target.mode !== target.mode) {
        rt.disconnect();
        this.onRemoteStream(null);
        this.dirty = true;
        return;
      }

      this.client = rt;
      this.session = {
        mode: target.mode,
        prompt,
        enhance: target.enhance,
        refKey: target.mode === 'edit' ? target.refKey : '',
      };
      rt.on('error', (err) => this.onError(`AI error: ${messageOf(err)}`));
      rt.on('sessionEnded', () => {
        if (this.client !== rt) return;
        this.client = null;
        this.session = null;
        this.setState('ended');
      });
      this.handleConnection(rt, rt.getConnectionState());
    } catch (err) {
      this.connecting = false;
      this.client = null;
      this.session = null;
      this.setState('error');
      this.onError(`Could not start AI: ${messageOf(err)}`);
    }
  }

  handleConnection(rt, state) {
    // Ignore events from sessions we already replaced or closed.
    const current = rt === null ? this.connecting : rt === this.client;
    if (!current || !state) return;
    // connecting | connected | generating | disconnected | reconnecting
    this.setState(state);
  }

  disconnect() {
    const rt = this.client;
    this.client = null;
    this.session = null;
    if (rt) {
      try {
        rt.disconnect();
      } catch {
        // already closed
      }
      this.onRemoteStream(null);
    }
  }
}

function messageOf(err) {
  if (!err) return 'unknown error';
  if (typeof err === 'string') return err;
  return err.message || err.code || String(err);
}
