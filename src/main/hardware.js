'use strict';

// Works out whether this computer can run the local face swap well.
const os = require('node:os');
const { execFile } = require('node:child_process');

const MIN_VRAM_MB = 2048; // the local models need about 2 GB of graphics memory
const MIN_WIN_BUILD = 18362; // DirectML needs Windows 10 version 1903 or newer

// Keeps whatever the command printed even when it ends with an error: PowerShell reports a failure
// when one registry key is off-limits to normal accounts, but still prints the rest.
function run(cmd, args, timeout = 25000) {
  return new Promise((resolve) => {
    execFile(cmd, args, { timeout, windowsHide: true, maxBuffer: 1024 * 1024 }, (_err, stdout) => {
      resolve(String(stdout || ''));
    });
  });
}

function parseJsonList(out) {
  const text = String(out || '').trim();
  const start = text.search(/[[{]/);
  if (start < 0) return [];
  try {
    const parsed = JSON.parse(text.slice(start));
    return Array.isArray(parsed) ? parsed : [parsed];
  } catch {
    return [];
  }
}

function powershell(script) {
  const exe = process.env.SystemRoot
    ? `${process.env.SystemRoot}\\System32\\WindowsPowerShell\\v1.0\\powershell.exe`
    : 'powershell.exe';
  return run(exe, ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script]);
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
    "  if ($p.DriverDesc) { [pscustomobject]@{ name = $p.DriverDesc; bytes = if ($q) { [double]$q } elseif ($m) { [double]$m } else { 0 } } }",
    "} | ConvertTo-Json -Compress",
    'exit 0',
  ].join('\n');
  return parseJsonList(await powershell(script))
    .filter((g) => g && g.name)
    .map((g) => ({ name: String(g.name), vramMB: Math.round(Number(g.bytes || 0) / (1024 * 1024)) }));
}

// Second opinion from WMI (its memory figure stops at 4 GB, which is enough to judge by).
async function windowsGpusWmi() {
  const script = [
    'Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | ForEach-Object {',
    '  [pscustomobject]@{ name = $_.Name; bytes = [double]($_.AdapterRAM) }',
    '} | ConvertTo-Json -Compress',
    'exit 0',
  ].join('\n');
  return parseJsonList(await powershell(script))
    .filter((g) => g && g.name)
    .map((g) => ({ name: String(g.name), vramMB: Math.round(Number(g.bytes || 0) / (1024 * 1024)) }));
}

const VENDORS = { 0x10de: 'NVIDIA', 0x1002: 'AMD', 0x8086: 'Intel', 0x1414: 'Microsoft Basic' };

// Last resort: the graphics chips Chromium sees (names only, no memory figure).
async function chromiumGpus(electronApp) {
  if (!electronApp) return [];
  try {
    const info = await electronApp.getGPUInfo('basic');
    return ((info && info.gpuDevice) || []).map((d) => ({
      name: d.deviceString || `${VENDORS[d.vendorId] || 'Unknown'} graphics (${Number(d.deviceId).toString(16)})`,
      vramMB: 0,
    }));
  } catch {
    return [];
  }
}

// Merge lists by name, keeping the largest memory figure for each card.
function mergeGpus(...lists) {
  const byName = new Map();
  for (const list of lists) {
    for (const g of list) {
      const key = g.name.trim().toLowerCase();
      const prev = byName.get(key);
      if (!prev || g.vramMB > prev.vramMB) byName.set(key, { name: g.name.trim(), vramMB: g.vramMB });
    }
  }
  return [...byName.values()];
}

function isIntegrated(name) {
  return /intel\(r\) (uhd|hd|iris)|intel (uhd|hd|iris)|^intel graphics|microsoft basic|microsoft remote|radeon\(tm\) graphics$|vega \d+ graphics/i.test(name);
}

function isDedicatedBrand(name) {
  return /nvidia|geforce|quadro|rtx|gtx|radeon (rx|pro)|amd radeon|\barc\b/i.test(name);
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
    const [registry, wmi] = await Promise.all([windowsGpus(), windowsGpusWmi()]);
    result.gpus = mergeGpus(registry, wmi);
    if (!result.gpus.length) result.gpus = await chromiumGpus(electronApp);
    result.sources = { registry: registry.length, wmi: wmi.length };
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
      // WMI caps memory at 4 GB, so a 4 GB card can read as 3.99.
      result.verdict = result.best.vramMB >= 4000 ? 'great' : 'ok';
      result.message = `${result.best.name} with ${(result.best.vramMB / 1024).toFixed(1)} GB can run the local face swap.`;
    } else if (result.best && !isIntegrated(result.best.name) && !result.best.vramMB && isDedicatedBrand(result.best.name)) {
      result.verdict = 'ok';
      result.message = `${result.best.name} found. Its memory size couldn't be read, but it should run the local face swap.`;
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
