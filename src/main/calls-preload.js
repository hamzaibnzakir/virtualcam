'use strict';

// Preload for the Calls window (WhatsApp Web, Meet, Zoom, Discord, Telegram... running inside the app).
// It replaces the page's camera and microphone with Brainbox's picture and voice. They arrive from the
// studio over a local WebRTC link (studio -> this window), so no driver is needed. Pages see one
// camera ("Brainbox VirtualCam") and one microphone ("Brainbox Voice"); speakers are left alone.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('__brainboxBridge', {
  signal: (data) => ipcRenderer.send('call:signal', JSON.parse(JSON.stringify(data))),
  onSignal: (handler) => ipcRenderer.on('call:signal', (_e, data) => handler(data)),
});

function injected() {
  const bridge = window.__brainboxBridge;
  if (!bridge || !navigator.mediaDevices || window.__brainboxInstalled) return;
  window.__brainboxInstalled = true;
  const md = navigator.mediaDevices;
  const realEnumerate = md.enumerateDevices.bind(md);
  let pc = null;
  let video = null;
  let audio = null;
  let waiters = [];

  const ready = () => video && audio && video.readyState === 'live' && audio.readyState === 'live';
  const flush = () => {
    if (!ready()) return;
    waiters.forEach((w) => w());
    waiters = [];
  };

  function connect() {
    if (pc && pc.connectionState !== 'failed' && pc.connectionState !== 'closed') return;
    video = null;
    audio = null;
    pc = new RTCPeerConnection({ iceServers: [] });
    pc.ontrack = (e) => {
      if (e.track.kind === 'video') video = e.track;
      else audio = e.track;
      e.track.addEventListener('ended', () => {
        if (e.track === video) video = null;
        if (e.track === audio) audio = null;
      });
      flush();
    };
    pc.onicecandidate = (e) => {
      if (e.candidate) bridge.signal({ type: 'candidate', candidate: e.candidate.toJSON() });
    };
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed') setTimeout(connect, 1000);
    };
    bridge.signal({ type: 'hello' });
  }

  bridge.onSignal(async (msg) => {
    try {
      if (msg.type === 'offer') {
        if (!pc) connect();
        await pc.setRemoteDescription({ type: 'offer', sdp: msg.sdp });
        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        bridge.signal({ type: 'answer', sdp: answer.sdp });
      } else if (msg.type === 'candidate' && pc) {
        await pc.addIceCandidate(msg.candidate);
      } else if (msg.type === 'restart') {
        if (pc) pc.close();
        pc = null;
        connect();
      }
    } catch (err) {
      console.warn('Brainbox link:', err && err.message);
    }
  });

  const waitReady = (ms) =>
    new Promise((resolve, reject) => {
      if (ready()) return resolve();
      const timer = setTimeout(() => reject(new DOMException('Brainbox VirtualCam is not sending yet. Is the studio running?', 'NotReadableError')), ms);
      waiters.push(() => {
        clearTimeout(timer);
        resolve();
      });
    });

  md.getUserMedia = async function getUserMedia(constraints = {}) {
    connect();
    await waitReady(25000);
    const out = new MediaStream();
    if (constraints.video) out.addTrack(video.clone());
    if (constraints.audio) out.addTrack(audio.clone());
    return out;
  };
  // Camera and microphone always count as allowed here (they're Brainbox's, not the real devices).
  if (navigator.permissions && navigator.permissions.query) {
    const realQuery = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = (desc) =>
      desc && (desc.name === 'camera' || desc.name === 'microphone')
        ? Promise.resolve({ state: 'granted', status: 'granted', name: desc.name, onchange: null, addEventListener() {}, removeEventListener() {} })
        : realQuery(desc);
  }
  if (navigator.getUserMedia) {
    navigator.getUserMedia = (c, ok, fail) => md.getUserMedia(c).then(ok, fail);
  }

  md.enumerateDevices = async function enumerateDevices() {
    const real = await realEnumerate().catch(() => []);
    const speakers = real.filter((d) => d.kind === 'audiooutput');
    const fake = (kind, deviceId, label) => ({
      deviceId,
      kind,
      label,
      groupId: 'brainbox',
      toJSON() {
        return { deviceId, kind, label, groupId: 'brainbox' };
      },
    });
    return [fake('videoinput', 'brainbox-camera', 'Brainbox VirtualCam'), fake('audioinput', 'brainbox-voice', 'Brainbox Voice'), ...speakers];
  };
  connect();
}

// Runs in the page's own world before its scripts.
contextBridge.executeInMainWorld({ func: injected });
