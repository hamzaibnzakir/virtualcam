'use strict';

// Notices when another app starts using a camera, so the studio can ask whether to use Brainbox
// VirtualCam. Windows records camera use per app (Settings > Privacy > Camera keeps the same list):
//   HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\...
// An app is using a camera right now when its LastUsedTimeStop is 0. Apps reading our virtual camera
// may not show up there (it isn't a hardware camera); the camera bridge reports those instead.
const path = require('node:path');
const { execFile } = require('node:child_process');

const KEY = 'HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\CapabilityAccessManager\\ConsentStore\\webcam';
const POLL_MS = 3000;

function query() {
  return new Promise((resolve) => {
    execFile('reg.exe', ['query', KEY, '/s', '/v', 'LastUsedTimeStop'], { windowsHide: true, timeout: 8000, maxBuffer: 4 << 20 }, (err, stdout) => {
      resolve(err && !stdout ? '' : String(stdout));
    });
  });
}

// A readable name from a consent-store key: "C:#Program Files#Zoom#bin#Zoom.exe" -> "Zoom",
// "Microsoft.WindowsCamera_8wekyb3d8bbwe" -> "WindowsCamera".
function appName(keyName) {
  if (keyName.includes('#')) {
    const exe = keyName.split('#').pop();
    return exe.replace(/\.exe$/i, '');
  }
  const pkg = keyName.split('_')[0];
  return pkg.split('.').pop();
}

function parse(out) {
  const inUse = new Map(); // key -> name
  let current = null;
  for (const line of out.split(/\r?\n/)) {
    if (line.startsWith('HKEY_')) {
      current = line.trim();
      continue;
    }
    const m = line.match(/LastUsedTimeStop\s+REG_QWORD\s+(0x[0-9a-f]+)/i);
    if (m && current && BigInt(m[1]) === 0n) {
      const leaf = current.split('\\').pop();
      if (leaf === 'NonPackaged' || leaf === 'webcam') continue;
      inUse.set(current, appName(leaf));
    }
  }
  return inUse;
}

class CameraWatch {
  // onStart({ name, key }) is called when an app (not this one) starts using a camera.
  constructor({ onStart, ownExe = process.execPath }) {
    this.onStart = onStart;
    this.own = ownExe.toLowerCase().replace(/\\/g, '#');
    this.ownDir = path.dirname(ownExe).toLowerCase().replace(/\\/g, '#');
    this.seen = new Set();
    this.timer = null;
    this.first = true;
  }

  start() {
    if (process.platform !== 'win32' || this.timer) return;
    const tick = async () => {
      const inUse = parse(await query());
      for (const [key, name] of inUse) {
        const k = key.toLowerCase();
        if (k.includes(this.own) || k.includes(this.ownDir)) continue; // this app reading your webcam
        if (!this.seen.has(key) && !this.first) this.onStart({ name, key });
      }
      this.seen = new Set(inUse.keys());
      this.first = false; // apps already using a camera when we start aren't news
    };
    tick();
    this.timer = setInterval(tick, POLL_MS);
  }

  stop() {
    clearInterval(this.timer);
    this.timer = null;
  }
}

module.exports = { CameraWatch, parse, appName };
