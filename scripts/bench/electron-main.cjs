// Hidden Electron window for the Lite benchmark (test tool only, never shipped).
const { app, BrowserWindow } = require('electron');
const path = require('node:path');

// Machines without a graphics chip still get WebGL through Chromium's software renderer.
if (process.platform === 'linux' || process.env.BENCH_SOFTWARE_GL === '1') {
  app.commandLine.appendSwitch('use-angle', 'swiftshader');
  app.commandLine.appendSwitch('enable-unsafe-swiftshader');
}

app.whenReady().then(() => {
  const win = new BrowserWindow({
    width: 1280,
    height: 720,
    show: false,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      sandbox: false,
      backgroundThrottling: false,
    },
  });
  win.webContents.on('console-message', (event, legacyLevel, legacyMessage) => {
    const message = event.message ?? legacyMessage;
    if (!/^INFO: Created TensorFlow Lite|^W\d{4}|^I\d{4}/.test(message)) process.stderr.write(`[bench] ${message}\n`);
  });
  win.loadFile(path.join(process.env.BENCH_DIST, 'bench.html'));
  win.on('closed', () => app.quit());
});

app.on('window-all-closed', () => app.quit());
