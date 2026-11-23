// AudioWorklet for the trained AI voices. It talks straight to the voice engine process over a
// MessagePort: your microphone goes out in 20 ms chunks, the AI voice comes back in blocks and is
// played from a small buffer, so a busy page never causes gaps.
const CHUNK = 960; // 20 ms at 48 kHz
const PREBUFFER = 3840; // 80 ms in hand before playing (absorbs uneven delivery)
const MAX_BUFFER = 48000; // more than 1 s behind: drop the oldest so the delay doesn't grow

class AiVoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.engine = null;
    this.active = false;
    this.chunk = new Float32Array(CHUNK);
    this.filled = 0;
    this.queue = [];
    this.queued = 0;
    this.head = 0; // read position in queue[0]
    this.playing = false;
    this.underruns = 0;
    this.sent = 0;
    this.chunksOut = 0;
    this.chunksIn = 0;
    this.hadInput = false;
    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg && msg.type === 'active') {
        this.active = Boolean(msg.on);
        if (!this.active) this.flush();
      } else if (msg && msg.type === 'port' && msg.port) {
        if (this.engine) this.engine.close();
        this.engine = msg.port;
        this.engine.onmessage = (ev) => this.receive(ev.data);
        this.engine.start();
      } else if (msg && msg.type === 'flush') {
        this.flush();
      }
    };
  }

  flush() {
    this.queue = [];
    this.queued = 0;
    this.head = 0;
    this.playing = false;
    this.filled = 0;
  }

  receive(data) {
    if (!this.active) return;
    const pcm = data instanceof Float32Array ? data : new Float32Array(data);
    if (!pcm.length) return;
    this.chunksIn += 1;
    this.queue.push(pcm);
    this.queued += pcm.length;
    while (this.queued - this.head > MAX_BUFFER && this.queue.length > 1) {
      this.queued -= this.queue.shift().length;
      this.head = 0;
    }
  }

  process(inputs, outputs) {
    const input = inputs[0] && inputs[0][0];
    const output = outputs[0];
    if (input) this.hadInput = true;
    if (input && this.engine && this.active) {
      let i = 0;
      while (i < input.length) {
        const take = Math.min(CHUNK - this.filled, input.length - i);
        this.chunk.set(input.subarray(i, i + take), this.filled);
        this.filled += take;
        i += take;
        if (this.filled === CHUNK) {
          // Copied, not transferred: the engine's port (an Electron MessagePortMain) drops transferred buffers.
          this.engine.postMessage(this.chunk);
          this.chunksOut += 1;
          this.filled = 0;
        }
      }
    }
    if (!output || !output[0]) return true;
    const n = output[0].length;
    const available = this.queued - this.head;
    if (!this.playing && available >= PREBUFFER) this.playing = true;
    const ch = output[0];
    if (this.playing) {
      let w = 0;
      while (w < n && this.queue.length) {
        const cur = this.queue[0];
        const take = Math.min(n - w, cur.length - this.head);
        ch.set(cur.subarray(this.head, this.head + take), w);
        w += take;
        this.head += take;
        if (this.head >= cur.length) {
          this.queue.shift();
          this.queued -= cur.length;
          this.head = 0;
        }
      }
      if (w < n) {
        ch.fill(0, w);
        this.playing = false;
        this.underruns += 1;
      }
    } else {
      ch.fill(0);
    }
    for (let c = 1; c < output.length; c += 1) output[c].set(ch);
    this.sent += n;
    if (this.sent >= sampleRate) {
      this.sent = 0;
      this.port.postMessage({ buffered: Math.round(((this.queued - this.head) / sampleRate) * 1000), underruns: this.underruns, chunksOut: this.chunksOut, chunksIn: this.chunksIn, linked: Boolean(this.engine), active: this.active, input: this.hadInput });
    }
    return true;
  }
}

registerProcessor('brainbox-ai-voice', AiVoiceProcessor);
