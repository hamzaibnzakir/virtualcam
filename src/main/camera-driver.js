'use strict';

// The VirtualCam camera device on Windows: install/uninstall it, and stream frames into it.
const fs = require('node:fs');
const path = require('node:path');
const { execFile, spawn } = require('node:child_process');

const CLSID = '{9CF79DDE-DA6F-46F4-886A-4ED6ED6A99A4}';

function run(cmd, args, timeout = 15000) {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout, windowsHide: true }, (err, stdout, stderr) => {
      resolve({ code: err ? (typeof err.code === 'number' ? err.code : 1) : 0, stdout: String(stdout), stderr: String(stderr) });
    });
  });
}

class CameraDriver {
  constructor({ binDir, onState }) {
    this.binDir = binDir; // contains x64\softcam.dll, x64\vcam-bridge.exe, x86\softcam.dll
    this.onState = onState || (() => {});
    this.bridge = null;
    this.state = { supported: process.platform === 'win32', installed: false, running: false, inUse: false, message: '' };
    this.width = 0;
    this.height = 0;
    this.dropped = 0;
  }

  files() {
    return {
      dll64: path.join(this.binDir, 'x64', 'softcam.dll'),
      dll32: path.join(this.binDir, 'x86', 'softcam.dll'),
      bridge: path.join(this.binDir, 'x64', 'vcam-bridge.exe'),
    };
  }

  emit(patch) {
    this.state = { ...this.state, ...patch };
    this.onState(this.state);
  }

  async refresh() {
    if (!this.state.supported) {
      this.emit({ message: 'The VirtualCam camera device is Windows only for now. Use OBS with the output window.' });
      return this.state;
    }
    const f = this.files();
    if (!fs.existsSync(f.dll64) || !fs.existsSync(f.bridge)) {
      this.emit({ installed: false, message: 'Camera files are missing from this build.' });
      return this.state;
    }
    // Registered, and pointing at this copy of the app?
    const res = await run('reg.exe', ['query', `HKCR\\CLSID\\${CLSID}\\InprocServer32`, '/ve']);
    const registered = res.code === 0 && res.stdout.toLowerCase().includes(f.dll64.toLowerCase());
    this.emit({
      installed: registered,
      message: registered ? '' : res.code === 0 ? 'The camera points at an older copy of the app. Install it again.' : '',
    });
    return this.state;
  }

  // Registers (or removes) the camera for 64-bit and 32-bit apps. Windows asks for admin approval once.
  async setInstalled(install) {
    if (!this.state.supported) return this.state;
    const f = this.files();
    const flag = install ? '' : "'/u',";
    const esc = (p) => p.replace(/'/g, "''");
    // One elevated script registers both, so Windows asks for approval only once.
    const lines = [
      `$a = Start-Process -FilePath "$env:WINDIR\\System32\\regsvr32.exe" -ArgumentList '/s',${flag}'"${esc(f.dll64)}"' -Wait -PassThru`,
      'if ($a.ExitCode -ne 0) { exit $a.ExitCode }',
    ];
    if (fs.existsSync(f.dll32)) {
      lines.push(
        `$b = Start-Process -FilePath "$env:WINDIR\\SysWOW64\\regsvr32.exe" -ArgumentList '/s',${flag}'"${esc(f.dll32)}"' -Wait -PassThru`,
        'if ($b.ExitCode -ne 0) { exit $b.ExitCode }'
      );
    }
    lines.push('exit 0');
    const script = path.join(require('node:os').tmpdir(), `virtualcam-${install ? 'install' : 'uninstall'}-${process.pid}.ps1`);
    fs.writeFileSync(script, lines.join('\r\n'));
    if (!install) this.stop();
    const outer =
      `$p = Start-Process -FilePath powershell.exe -Verb RunAs -Wait -PassThru -WindowStyle Hidden ` +
      `-ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','"${esc(script)}"'; exit $p.ExitCode`;
    const res = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', outer], 120000);
    fs.rmSync(script, { force: true });
    await this.refresh();
    if (res.code !== 0 && install && !this.state.installed) {
      this.emit({ message: 'Camera install was cancelled or blocked. Admin approval is needed once.' });
    }
    return this.state;
  }

  start(width, height) {
    if (!this.state.supported || !this.state.installed) return false;
    if (this.bridge && this.width === width && this.height === height) return true;
    this.stop();
    this.width = width;
    this.height = height;
    const child = spawn(this.files().bridge, [String(width), String(height)], {
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'ignore'],
    });
    this.bridge = child;
    child.stdin.on('error', () => {}); // the bridge exiting mid-write is handled by 'exit'
    let buffered = '';
    child.stdout.on('data', (chunk) => {
      buffered += chunk.toString();
      let nl;
      while ((nl = buffered.indexOf('\n')) >= 0) {
        const line = buffered.slice(0, nl).trim();
        buffered = buffered.slice(nl + 1);
        if (line === 'ready') this.emit({ running: true, message: '' });
        else if (line === 'connected') this.emit({ inUse: true });
        else if (line === 'disconnected') this.emit({ inUse: false });
        else if (line.startsWith('error')) this.emit({ message: line.slice(6) });
      }
    });
    child.on('exit', (code) => {
      if (this.bridge !== child) return;
      this.bridge = null;
      this.emit({
        running: false,
        inUse: false,
        message: code === 2 ? 'Another copy of VirtualCam is already sending to the camera.' : this.state.message,
      });
    });
    return true;
  }

  stop() {
    const child = this.bridge;
    this.bridge = null;
    if (child) {
      try {
        child.stdin.end();
      } catch {
        // already gone
      }
      setTimeout(() => {
        if (child.exitCode === null) child.kill();
      }, 1500);
    }
    this.emit({ running: false, inUse: false });
  }

  // frame: BGR bytes, width*height*3. Frames are dropped while the camera is busy taking the last one.
  sendFrame(frame) {
    const child = this.bridge;
    if (!child || !this.state.running || child.stdin.writableNeedDrain) {
      this.dropped += 1;
      return false;
    }
    if (frame.byteLength !== this.width * this.height * 3) return false;
    child.stdin.write(Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength));
    return true;
  }
}

module.exports = { CameraDriver, CLSID };
