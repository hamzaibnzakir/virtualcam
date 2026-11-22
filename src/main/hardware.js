'use strict';

// Works out whether this computer can run the local face swap well.
const os = require('node:os');
const { execFile } = require('node:child_process');

const MIN_VRAM_MB = 2048; // the local models need about 2 GB of graphics memory
const MIN_WIN_BUILD = 18362; // DirectML needs Windows 10 version 1903 or newer

function run(cmd, args, timeout = 8000) {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout, windowsHide: true, maxBuffer: 1024 * 1024 }, (err, stdout) => {
      resolve(err ? '' : String(stdout));
    });
  });
}

// Graphics cards and their dedicated memory, read from the Windows registry
// (the WMI "AdapterRAM" value tops out at 4 GB, the registry value does not).
async function windowsGpus() {
  const script = [
    "$k='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}'",
    "Get-ChildItem $k -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\\d{4}$' } | ForEach-Object {",
    "  $p = Get-ItemProperty $_.PSPath -ErrorAction SilentlyContinue",
    "  $q = $p.'HardwareInformation.qwMemorySize'",
    "  $m = $p.'HardwareInformation.MemorySize'",
    "  if ($m -is [byte[]]) { $m = [BitConverter]::ToUInt32($m, 0) }",
    "  [pscustomobject]@{ name = $p.DriverDesc; bytes = if ($q) { [double]$q } elseif ($m) { [double]$m } else { 0 } }",
    "} | ConvertTo-Json -Compress",
  ].join('\n');
  const out = await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script]);
  try {
    const parsed = JSON.parse(out.trim() || '[]');
    const list = Array.isArray(parsed) ? parsed : [parsed];
    return list
      .filter((g) => g && g.name)
      .map((g) => ({ name: String(g.name), vramMB: Math.round(Number(g.bytes || 0) / (1024 * 1024)) }));
  } catch {
    return [];
  }
}

function isIntegrated(name) {
  return /intel\(r\) (uhd|hd|iris)|intel (uhd|hd|iris)|microsoft basic|radeon\(tm\) graphics$|vega \d+ graphics/i.test(name);
}

async function checkHardware(electronApp) {
  const platform = process.platform;
  const result = {
    platform,
    ramGB: Math.round(os.totalmem() / 1024 ** 3),
    gpus: [],
    best: null,
    verdict: 'unknown', // great | ok | cpu-only | unsupported | unknown
    message: '',
  };

  if (platform === 'win32') {
    const build = Number(os.release().split('.')[2] || 0);
    result.windowsBuild = build;
    result.gpus = await windowsGpus();
    if (build && build < MIN_WIN_BUILD) {
      result.verdict = 'unsupported';
      result.message = 'Local AI needs Windows 10 version 1903 or newer. Use the cloud AI instead.';
      return result;
    }
  } else if (electronApp) {
    try {
      const info = await electronApp.getGPUInfo('basic');
      const devices = (info && info.gpuDevice) || [];
      result.gpus = devices.map((d) => ({ name: d.deviceString || `GPU ${d.vendorId}:${d.deviceId}`, vramMB: 0 }));
    } catch {
      // no GPU info available
    }
  }

  const dedicated = result.gpus.filter((g) => !isIntegrated(g.name));
  result.best = dedicated.sort((a, b) => b.vramMB - a.vramMB)[0] || result.gpus[0] || null;

  if (platform === 'win32') {
    if (result.best && !isIntegrated(result.best.name) && result.best.vramMB >= MIN_VRAM_MB) {
      result.verdict = result.best.vramMB >= 4096 ? 'great' : 'ok';
      result.message = `${result.best.name} with ${(result.best.vramMB / 1024).toFixed(1)} GB can run the local face swap.`;
    } else if (result.best) {
      result.verdict = 'cpu-only';
      result.message = `${result.best.name} is below the 2 GB graphics memory the local AI needs. It will run, but slowly. The cloud AI is recommended.`;
    } else {
      result.verdict = 'cpu-only';
      result.message = 'No graphics card found. The local AI would run on the processor only, which is slow. The cloud AI is recommended.';
    }
  } else if (platform === 'darwin') {
    result.verdict = 'ok';
    result.message = 'Your Mac can run the local face swap.';
  } else {
    result.verdict = 'cpu-only';
    result.message = 'Local AI runs on the processor here, which is slow.';
  }
  return result;
}

module.exports = { checkHardware, MIN_VRAM_MB };
