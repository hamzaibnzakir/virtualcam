// Sends the studio's picture and voice to pages in the Calls window over local WebRTC (one link per
// page frame that asks). The page side is src/main/calls-preload.js.
export class CallLink {
  constructor({ api, videoTrack, audioTrack, onState = () => {} }) {
    this.api = api;
    this.videoTrack = videoTrack; // async () => MediaStreamTrack
    this.audioTrack = audioTrack; // async () => MediaStreamTrack
    this.onState = onState;
    this.peers = new Map(); // frame id -> RTCPeerConnection
    api.onCallSignal(({ id, data }) => this.handle(id, data));
    api.onCallClosed(() => this.closeAll());
  }

  get connected() {
    return [...this.peers.values()].some((pc) => pc.connectionState === 'connected');
  }

  async handle(id, data) {
    try {
      if (data.type === 'hello') await this.offer(id);
      else if (data.type === 'answer') await this.peers.get(id)?.setRemoteDescription({ type: 'answer', sdp: data.sdp });
      else if (data.type === 'candidate') await this.peers.get(id)?.addIceCandidate(data.candidate);
    } catch (err) {
      console.warn('Call link:', err && err.message);
    }
  }

  async offer(id) {
    const old = this.peers.get(id);
    if (old) old.close();
    const pc = new RTCPeerConnection({ iceServers: [] });
    this.peers.set(id, pc);
    const stream = new MediaStream();
    const video = await this.videoTrack();
    const audio = await this.audioTrack();
    pc.addTrack(video, stream);
    pc.addTrack(audio, stream);
    pc.onicecandidate = (e) => {
      if (e.candidate) {
        if (this.api.smoke) console.warn(`[link] local candidate ${e.candidate.candidate}`);
        this.api.callSignal(id, { type: 'candidate', candidate: e.candidate.toJSON() });
      }
    };
    pc.oniceconnectionstatechange = () => {
      if (this.api.smoke) console.warn(`[link] ice ${pc.iceConnectionState}`);
    };
    pc.onconnectionstatechange = () => {
      if (this.api.smoke) console.warn(`[link] connection ${pc.connectionState}`);
      if (pc.connectionState === 'connected') this.tune(pc);
      if (pc.connectionState === 'closed' || pc.connectionState === 'failed') this.peers.delete(id);
      this.onState();
    };
    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await this.api.callSignal(id, { type: 'offer', sdp: offer.sdp });
  }

  // On this computer there's bandwidth to spare: keep the picture sharp.
  async tune(pc) {
    for (const sender of pc.getSenders()) {
      if (!sender.track || sender.track.kind !== 'video') continue;
      const params = sender.getParameters();
      params.encodings = params.encodings && params.encodings.length ? params.encodings : [{}];
      params.encodings[0].maxBitrate = 8_000_000;
      params.encodings[0].maxFramerate = 30;
      params.degradationPreference = 'maintain-resolution';
      await sender.setParameters(params).catch(() => {});
    }
  }

  // A new voice track (voice changed, microphone switched on): swap it into every link.
  async replaceAudio(track) {
    for (const pc of this.peers.values()) {
      for (const sender of pc.getSenders()) if (sender.track && sender.track.kind === 'audio') await sender.replaceTrack(track);
    }
  }

  // The picture changed (Brainbox picture <-> your real camera): swap it into every link.
  async replaceVideo(track) {
    for (const pc of this.peers.values()) {
      for (const sender of pc.getSenders()) if (sender.track && sender.track.kind === 'video') await sender.replaceTrack(track);
    }
  }

  closeAll() {
    for (const pc of this.peers.values()) pc.close();
    this.peers.clear();
    this.onState();
  }
}
