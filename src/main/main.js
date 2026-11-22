'use strict';

const {
  app,
  BrowserWindow,
  desktopCapturer,
  dialog,
  globalShortcut,
  ipcMain,
  safeStorage,
  screen,
  session,
  shell,
} = require('electron');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const SMOKE = process.env.VCAM_SMOKE === '1';
const DIST = path.join(__dirname, '..', '..', 'dist');
const PRELOAD = path.join(__dirname, 'preload.js');
const MAX_FILE_BYTES = 512 * 1024 * 1024;

if (SMOKE) {
  // Headless test run: fake camera, auto-accept prompts, throwaway profile.
  app.commandLine.appendSwitch('use-fake-device-for-media-stream');
  app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
  app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'vcam-smoke-')));
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
}

let mainWindow = null;
let switcherWindow = null;
let sceneSnapshot = { scenes: [], activeId: null };
const hotkeyFailures = [];
const smokeLog = [];

// ---------- files in userData ----------

function dataFile(name) {
  return path.join(app.getPath('userData'), name);
}

function readJson(name) {
  try {
    return JSON.parse(fs.readFileSync(dataFile(name), 'utf8'));
  } catch {
    return null;
  }
}

function writeJson(name, value) {
  const target = dataFile(name);
  const tmp = `${target}.tmp`;
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(tmp, JSON.stringify(value, null, 2));
  fs.renameSync(tmp, target);
}

function getSecret(name) {
  const all = readJson('secrets.json') || {};
  const entry = all[name];
  if (!entry) return '';
  try {
    if (entry.enc && safeStorage.isEncryptionAvailable()) {
      return safeStorage.decryptString(Buffer.from(entry.value, 'base64'));
    }
    return entry.enc ? '' : entry.value;
  } catch {
    return '';
  }
}

function setSecret(name, value) {
  const all = readJson('secrets.json') || {};
  if (!value) {
    delete all[name];
  } else if (safeStorage.isEncryptionAvailable()) {
    all[name] = { enc: true, value: safeStorage.encryptString(value).toString('base64') };
  } else {
    all[name] = { enc: false, value };
  }
  writeJson('secrets.json', all);
  return { encrypted: safeStorage.isEncryptionAvailable() };
}

// ---------- windows ----------

function isOwnPage(url) {
  return url.startsWith('file://') || url === 'about:blank';
}

function hardenContents(contents) {
  contents.on('will-navigate', (event, url) => {
    if (!isOwnPage(url)) event.preventDefault();
  });
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 940,
    minWidth: 980,
    minHeight: 640,
    title: 'VirtualCam',
    backgroundColor: '#0B0C10',
    autoHideMenuBar: true,
    show: !SMOKE || Boolean(process.env.VCAM_SMOKE_SHOT),
    webPreferences: {
      preload: PRELOAD,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      // Keep rendering frames while minimized: the output must never freeze mid-call.
      backgroundThrottling: false,
    },
  });

  hardenContents(mainWindow.webContents);

  // The studio opens a plain about:blank window and draws the output into it,
  // so OBS (or a capture tool) can grab a clean 1280x720 frame.
  mainWindow.webContents.setWindowOpenHandler(({ url, frameName }) => {
    if (url === 'about:blank' && frameName === 'vcam-output') {
      return {
        action: 'allow',
        overrideBrowserWindowOptions: {
          width: 1280,
          height: 720,
          useContentSize: true,
          title: 'VirtualCam Output',
          backgroundColor: '#000000',
          autoHideMenuBar: true,
          show: !SMOKE,
          webPreferences: { backgroundThrottling: false },
        },
      };
    }
    if (/^https:\/\//.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.webContents.on('did-create-window', (child) => {
    hardenContents(child.webContents);
    child.setMenuBarVisibility(false);
  });

  if (SMOKE) {
    mainWindow.webContents.on('console-message', (event, legacyLevel, legacyMessage) => {
      const level = event.level ?? ['verbose', 'info', 'warning', 'error'][legacyLevel] ?? 'info';
      const message = event.message ?? legacyMessage;
      smokeLog.push({ level, message });
    });
    mainWindow.webContents.on('render-process-gone', (_e, details) => {
      smokeLog.push({ level: 'error', message: `renderer gone: ${details.reason}` });
    });
  }

  mainWindow.loadFile(path.join(DIST, 'studio.html'));
  mainWindow.on('closed', () => {
    mainWindow = null;
    // Closing the studio quits everything, including the output window.
    app.quit();
  });
}

function sendToStudio(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, payload);
}

function showSwitcher() {
  if (!switcherWindow || switcherWindow.isDestroyed()) {
    switcherWindow = new BrowserWindow({
      width: 440,
      height: 460,
      frame: false,
      resizable: false,
      movable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      show: false,
      transparent: true,
      backgroundColor: '#00000000',
      webPreferences: {
        preload: PRELOAD,
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    });
    hardenContents(switcherWindow.webContents);
    switcherWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    if (SMOKE) {
      switcherWindow.webContents.on('console-message', (event, legacyLevel, legacyMessage) => {
        const level = event.level ?? ['verbose', 'info', 'warning', 'error'][legacyLevel] ?? 'info';
        smokeLog.push({ level, message: `[switcher] ${event.message ?? legacyMessage}` });
      });
      switcherWindow.webContents.once('did-finish-load', () => {
        setTimeout(() => {
          switcherWindow.webContents
            .executeJavaScript('document.querySelectorAll("#list button").length')
            .then((count) => smokeLog.push({ level: 'info', message: `[switcher] rows=${count}` }))
            .catch((err) => smokeLog.push({ level: 'error', message: `[switcher] ${err.message}` }));
        }, 500);
      });
    }
    switcherWindow.on('blur', () => hideSwitcher());
    switcherWindow.loadFile(path.join(DIST, 'switcher.html'));
    switcherWindow.webContents.once('did-finish-load', () => presentSwitcher());
    return;
  }
  presentSwitcher();
}

function presentSwitcher() {
  if (!switcherWindow || switcherWindow.isDestroyed()) return;
  const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
  const { x, y, width } = display.workArea;
  const [w] = switcherWindow.getSize();
  switcherWindow.setPosition(Math.round(x + (width - w) / 2), Math.round(y + 80));
  switcherWindow.webContents.send('switcher:scenes', sceneSnapshot);
  switcherWindow.show();
  switcherWindow.focus();
}

function hideSwitcher() {
  if (switcherWindow && !switcherWindow.isDestroyed() && switcherWindow.isVisible()) {
    switcherWindow.hide();
  }
}

// ---------- hotkeys ----------

function registerHotkeys() {
  const bind = (accelerator, fn) => {
    let ok = false;
    try {
      ok = globalShortcut.register(accelerator, fn);
    } catch {
      ok = false;
    }
    if (!ok) hotkeyFailures.push(accelerator);
  };

  for (let i = 1; i <= 9; i += 1) {
    bind(`CommandOrControl+Alt+${i}`, () => sendToStudio('hotkey', { action: 'scene', index: i - 1 }));
  }
  bind('CommandOrControl+Alt+0', () => sendToStudio('hotkey', { action: 'panic' }));
  bind('CommandOrControl+Alt+Space', () => showSwitcher());
}

// ---------- IPC ----------

function registerIpc() {
  ipcMain.handle('state:load', () => readJson('state.json'));
  ipcMain.handle('state:save', (_e, state) => {
    if (state && typeof state === 'object') writeJson('state.json', state);
    return true;
  });

  ipcMain.handle('secret:get', (_e, name) => (typeof name === 'string' ? getSecret(name) : ''));
  ipcMain.handle('secret:set', (_e, name, value) => {
    if (typeof name !== 'string') return { encrypted: false };
    return setSecret(name, typeof value === 'string' ? value.trim() : '');
  });

  ipcMain.handle('file:pick', async (_e, kind) => {
    const filters =
      kind === 'video'
        ? [{ name: 'Video', extensions: ['mp4', 'webm', 'mov', 'm4v'] }]
        : [{ name: 'Image', extensions: ['png', 'jpg', 'jpeg', 'webp', 'gif'] }];
    const result = await dialog.showOpenDialog(mainWindow, { properties: ['openFile'], filters });
    return result.canceled || !result.filePaths[0] ? null : result.filePaths[0];
  });

  ipcMain.handle('file:read', async (_e, filePath) => {
    if (typeof filePath !== 'string' || !path.isAbsolute(filePath)) throw new Error('Invalid file path');
    const stat = await fs.promises.stat(filePath);
    if (!stat.isFile()) throw new Error('Not a file');
    if (stat.size > MAX_FILE_BYTES) throw new Error('File is larger than 512 MB');
    const data = await fs.promises.readFile(filePath);
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  });

  ipcMain.handle('hotkeys:failures', () => hotkeyFailures.slice());

  ipcMain.on('scenes:publish', (_e, snapshot) => {
    if (!snapshot || !Array.isArray(snapshot.scenes)) return;
    sceneSnapshot = snapshot;
    if (switcherWindow && !switcherWindow.isDestroyed()) {
      switcherWindow.webContents.send('switcher:scenes', sceneSnapshot);
    }
  });

  ipcMain.on('switcher:choose', (_e, index) => {
    if (Number.isInteger(index)) sendToStudio('hotkey', { action: 'scene', index });
    hideSwitcher();
  });
  ipcMain.on('switcher:panic', () => {
    sendToStudio('hotkey', { action: 'panic' });
    hideSwitcher();
  });
  ipcMain.on('switcher:close', () => hideSwitcher());

  ipcMain.on('smoke:report', async (_e, report) => {
    if (!SMOKE) return;
    const out = process.env.VCAM_SMOKE_OUT || path.join(os.tmpdir(), 'vcam-smoke.json');
    fs.writeFileSync(out, JSON.stringify({ report, console: smokeLog, hotkeyFailures }, null, 2));
    if (process.env.VCAM_SMOKE_SHOT && mainWindow) {
      const image = await mainWindow.webContents.capturePage();
      fs.writeFileSync(process.env.VCAM_SMOKE_SHOT, image.toPNG());
    }
    app.quit();
  });
}

// ---------- app lifecycle ----------

app.on('second-instance', () => {
  if (!mainWindow) return;
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
});

app.whenReady().then(() => {
  const ses = session.defaultSession;

  ses.setPermissionRequestHandler((webContents, permission, callback) => {
    const allowed = ['media', 'display-capture', 'clipboard-sanitized-write'];
    callback(allowed.includes(permission) && isOwnPage(webContents.getURL()));
  });

  // "Screen" scenes capture the screen the cursor is on, with no picker.
  ses.setDisplayMediaRequestHandler(async (_request, callback) => {
    try {
      const sources = await desktopCapturer.getSources({ types: ['screen'] });
      const cursorDisplay = screen.getDisplayNearestPoint(screen.getCursorScreenPoint());
      const match = sources.find((s) => s.display_id === String(cursorDisplay.id)) || sources[0];
      if (match) callback({ video: match });
      else callback({});
    } catch {
      callback({});
    }
  });

  registerIpc();
  registerHotkeys();
  createMainWindow();

  if (SMOKE) {
    setTimeout(() => showSwitcher(), 4000);
    // Safety net: never hang a test run.
    setTimeout(() => {
      const out = process.env.VCAM_SMOKE_OUT || path.join(os.tmpdir(), 'vcam-smoke.json');
      fs.writeFileSync(out, JSON.stringify({ report: { timeout: true }, console: smokeLog, hotkeyFailures }, null, 2));
      app.quit();
    }, 70000);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  app.quit();
});
