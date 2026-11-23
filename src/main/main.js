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
  MessageChannelMain,
  Menu,
  nativeImage,
  Notification,
  session,
  shell,
  Tray,
  utilityProcess,
} = require('electron');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { checkHardware } = require('./hardware');
const { Downloader, status: modelStatus } = require('./models');
const { CameraDriver } = require('./camera-driver');
const { CameraWatch } = require('./camera-watch');
const { setupUpdater } = require('./updater');

const SMOKE = process.env.VCAM_SMOKE === '1';
const DIST = path.join(__dirname, '..', '..', 'dist');
const PRELOAD = path.join(__dirname, 'preload.js');
const MAX_FILE_BYTES = 512 * 1024 * 1024;

// The studio and the Calls window talk over a local WebRTC link. Chromium normally hides local
// addresses behind .local names that need a working mDNS service to resolve, which isn't always the
// case (blocked by firewalls or VPNs), so the link would never connect. Show plain local addresses.
app.commandLine.appendSwitch('disable-features', 'WebRtcHideLocalIpsWithMdns');

if (SMOKE) {
  // Headless test run: fake camera (unless testing real devices), auto-accept prompts, throwaway profile.
  if (process.env.VCAM_SMOKE_REAL_DEVICES !== '1') app.commandLine.appendSwitch('use-fake-device-for-media-stream');
  // Test machines without a graphics chip still get WebGL through Chromium's software renderer.
  if (process.platform === 'linux' || process.env.VCAM_SMOKE_SOFTWARE_GL === '1') {
    app.commandLine.appendSwitch('use-angle', 'swiftshader');
    app.commandLine.appendSwitch('enable-unsafe-swiftshader');
  }
  app.commandLine.appendSwitch('use-fake-ui-for-media-stream');
  app.setPath('userData', fs.mkdtempSync(path.join(os.tmpdir(), 'vcam-smoke-')));
} else if (!process.env.VCAM_USER_DATA_NEW) {
  // Versions before 1.1 were called VirtualCam: keep using that folder, so scenes, photos and the
  // downloaded AI models carry over to Brainbox VirtualCam.
  app.setPath('userData', path.join(app.getPath('appData'), 'VirtualCam'));
}

if (!app.requestSingleInstanceLock()) {
  app.quit();
}

// Started by Windows at sign-in: run in the tray without opening the studio window.
const START_HIDDEN = process.argv.includes('--background');
const BRAND = path.join(__dirname, '..', '..', 'assets', 'brand');
let tray = null;
let promptWindow = null;
let updater = null;
let cameraWatch = null;
// Preferences the studio sends (settings dialog); defaults until it does.
let prefs = { runInBackground: true, startWithWindows: true, askOnCameraUse: true };
let toldAboutTray = false;
let callWindow = null;

let mainWindow = null;
let switcherWindow = null;
let sceneSnapshot = { scenes: [], activeId: null };
const hotkeyFailures = [];
const smokeLog = [];
let faceWorker = null;
let voiceWorker = null;
let hardwareCache = null;
let downloader = null;
let camera = null;

function modelsDir() {
  return process.env.VCAM_MODELS_DIR || path.join(app.getPath('userData'), 'models');
}

// The face engine's log, started fresh when it grows past 1 MB.
function engineLogFile() {
  const dir = app.getPath('logs');
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, 'face-engine.log');
  try {
    if (fs.statSync(file).size > 1024 * 1024) fs.renameSync(file, path.join(dir, 'face-engine.old.log'));
  } catch {
    // no log yet
  }
  return file;
}

// Camera files ship in resources/vendor/win when packaged, and in the repo's vendor/win in development.
function vendorDir() {
  const packaged = path.join(process.resourcesPath || '', 'vendor', 'win');
  return app.isPackaged ? packaged : path.join(__dirname, '..', '..', 'vendor', 'win');
}

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
    title: 'Brainbox VirtualCam',
    backgroundColor: '#0B0C10',
    autoHideMenuBar: true,
    show: (!SMOKE && !START_HIDDEN) || Boolean(process.env.VCAM_SMOKE_SHOT),
    icon: path.join(BRAND, 'icon.png'),
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
          title: 'Brainbox VirtualCam Output',
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

  // Closing the window keeps the app (and the camera) running in the tray, unless you turned that off.
  mainWindow.on('close', (event) => {
    if (app.isQuitting || SMOKE || !prefs.runInBackground) return;
    event.preventDefault();
    mainWindow.hide();
    if (!toldAboutTray && Notification.isSupported()) {
      toldAboutTray = true;
      new Notification({
        title: 'Brainbox VirtualCam is still running',
        body: 'The camera keeps working and will ask you when an app starts using it. Quit from the tray icon.',
        icon: path.join(BRAND, 'icon.png'),
      }).show();
    }
  });
  mainWindow.on('closed', () => {
    mainWindow = null;
    // Closing the studio quits everything, including the output window.
    app.quit();
  });
}

function sendToStudio(channel, payload) {
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send(channel, payload);
}

// ---------- Calls: web call apps inside the app, fed Brainbox's picture and voice ----------
const CALLS_PRELOAD = path.join(__dirname, 'calls-preload.js');
const CALL_SITES = {
  whatsapp: 'https://web.whatsapp.com/',
  meet: 'https://meet.google.com/',
  zoom: 'https://app.zoom.us/wc',
  discord: 'https://discord.com/app',
  telegram: 'https://web.telegram.org/',
  messenger: 'https://www.messenger.com/',
  teams: 'https://teams.live.com/',
};

function callSession() {
  const ses = session.fromPartition('persist:calls');
  if (!ses.brainboxReady) {
    ses.brainboxReady = true;
    // Look like plain Chrome: some sites refuse browsers they don't know.
    ses.setUserAgent(ses.getUserAgent().replace(/\s(Electron|brainbox-virtualcam|virtualcam|Brainbox VirtualCam)\/\S+/gi, ''));
    ses.setPermissionRequestHandler((_wc, permission, callback) => {
      callback(['media', 'notifications', 'clipboard-sanitized-write', 'fullscreen', 'clipboard-read'].includes(permission));
    });
  }
  return ses;
}

function openCall(target) {
  const url = CALL_SITES[target] || (SMOKE && target === 'smoke-test' ? `file://${path.join(DIST, 'call-test.html')}` : null)
    || (typeof target === 'string' && /^https:\/\/[^\s]+$/.test(target) ? target : null);
  if (!url) return false;
  if (callWindow && !callWindow.isDestroyed()) {
    callWindow.loadURL(url);
    callWindow.show();
    callWindow.focus();
    return true;
  }
  callWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    title: 'Brainbox VirtualCam · Calls',
    backgroundColor: '#0B0C10',
    autoHideMenuBar: true,
    icon: path.join(BRAND, 'icon.png'),
    show: !SMOKE,
    webPreferences: {
      session: callSession(),
      preload: CALLS_PRELOAD,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      nodeIntegrationInSubFrames: true, // the camera/mic swap also runs inside embedded call frames
      backgroundThrottling: false,
    },
  });
  callWindow.webContents.setWindowOpenHandler(({ url: next }) => {
    // Pop-ups of the same site (meeting windows, sign-in) stay inside; other links go to the browser.
    try {
      const host = new URL(next).host;
      if (host && host === new URL(callWindow.webContents.getURL()).host) return { action: 'allow' };
    } catch {
      // fall through
    }
    if (/^https:\/\//.test(next)) shell.openExternal(next);
    return { action: 'deny' };
  });
  callWindow.on('closed', () => {
    callWindow = null;
    sendToStudio('call:closed', {});
  });
  if (SMOKE) {
    callWindow.webContents.on('console-message', (event, legacyLevel, legacyMessage) => {
      smokeLog.push({ level: 'warning', message: `[calls] ${event.message ?? legacyMessage}` });
    });
    callWindow.webContents.on('preload-error', (_e, file, err) => smokeLog.push({ level: 'error', message: `[calls] preload ${file}: ${err.message}` }));
  }
  callWindow.loadURL(url);
  return true;
}

function openStudio() {
  if (!mainWindow || mainWindow.isDestroyed()) createMainWindow();
  if (mainWindow.isMinimized()) mainWindow.restore();
  mainWindow.show();
  mainWindow.focus();
}

function quitApp() {
  app.isQuitting = true;
  app.quit();
}

function createTray() {
  if (tray || SMOKE) return;
  const image = nativeImage.createFromPath(path.join(BRAND, 'tray.png'));
  tray = new Tray(image);
  tray.setToolTip('Brainbox VirtualCam');
  const menu = () =>
    Menu.buildFromTemplate([
      { label: 'Open the studio', click: openStudio },
      { label: 'Scene switcher', click: () => showSwitcher() },
      { type: 'separator' },
      {
        label: 'Ask when an app starts using a camera',
        type: 'checkbox',
        checked: prefs.askOnCameraUse,
        click: (item) => sendToStudio('prefs:set', { askOnCameraUse: item.checked }),
      },
      {
        label: 'Start with Windows',
        type: 'checkbox',
        checked: prefs.startWithWindows,
        click: (item) => sendToStudio('prefs:set', { startWithWindows: item.checked }),
      },
      { type: 'separator' },
      { label: 'Quit Brainbox VirtualCam', click: quitApp },
    ]);
  tray.setContextMenu(menu());
  tray.on('click', openStudio);
  tray.rebuild = () => tray.setContextMenu(menu());
}

function applyPrefs(next) {
  prefs = { ...prefs, ...next };
  if (app.isPackaged && !SMOKE && (process.platform === 'win32' || process.platform === 'darwin')) {
    app.setLoginItemSettings({ openAtLogin: Boolean(prefs.startWithWindows), args: ['--background'] });
  }
  if (tray && tray.rebuild) tray.rebuild();
}

// A small card in the corner of the screen, above every window, with a question and buttons.
// payload: { title, body, actions: [{ id, label, primary }], timeoutMs }
function showPrompt(payload) {
  const { workArea } = screen.getPrimaryDisplay();
  const width = 400;
  const height = 190;
  if (!promptWindow || promptWindow.isDestroyed()) {
    promptWindow = new BrowserWindow({
      width,
      height,
      x: workArea.x + workArea.width - width - 16,
      y: workArea.y + workArea.height - height - 16,
      frame: false,
      resizable: false,
      skipTaskbar: true,
      alwaysOnTop: true,
      show: false,
      transparent: true,
      backgroundColor: '#00000000',
      webPreferences: { preload: PRELOAD, contextIsolation: true, nodeIntegration: false, sandbox: true },
    });
    hardenContents(promptWindow.webContents);
    promptWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    promptWindow.setAlwaysOnTop(true, 'screen-saver');
    promptWindow.loadFile(path.join(DIST, 'prompt.html'));
    promptWindow.once('ready-to-show', () => {
      promptWindow.webContents.send('prompt:data', promptWindow.payload);
      promptWindow.showInactive();
    });
  } else {
    promptWindow.webContents.send('prompt:data', payload);
    promptWindow.showInactive();
  }
  promptWindow.payload = payload;
  clearTimeout(promptWindow.timer);
  promptWindow.timer = setTimeout(() => hidePrompt(), payload.timeoutMs || 45000);
}

function hidePrompt() {
  if (promptWindow && !promptWindow.isDestroyed()) promptWindow.hide();
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
  // App preferences (tray, start with Windows, ask on camera use), owned by the studio's settings.
  ipcMain.handle('prefs:apply', (_e, next) => {
    if (next && typeof next === 'object') {
      applyPrefs({
        runInBackground: next.runInBackground !== false,
        startWithWindows: next.startWithWindows !== false,
        askOnCameraUse: next.askOnCameraUse !== false,
      });
    }
    return prefs;
  });
  // The corner question card: the studio decides what to ask; clicks go back to the studio.
  ipcMain.handle('prompt:show', (_e, payload) => {
    if (!payload || typeof payload !== 'object') return false;
    const clean = {
      title: String(payload.title || '').slice(0, 120),
      body: String(payload.body || '').slice(0, 400),
      actions: (Array.isArray(payload.actions) ? payload.actions : []).slice(0, 3).map((a) => ({
        id: String(a.id).slice(0, 40),
        label: String(a.label).slice(0, 40),
        primary: Boolean(a.primary),
      })),
      timeoutMs: Math.min(120000, Math.max(5000, Number(payload.timeoutMs) || 45000)),
    };
    showPrompt(clean);
    return true;
  });
  ipcMain.handle('prompt:action', (e, id) => {
    if (!promptWindow || e.sender !== promptWindow.webContents) return false;
    hidePrompt();
    if (id === 'open') openStudio();
    sendToStudio('prompt:action', String(id));
    return true;
  });
  // Calls: open a site; relay the WebRTC link messages between call pages (any frame) and the studio.
  ipcMain.handle('calls:open', (_e, target) => openCall(target));
  ipcMain.on('call:signal', (e, data) => {
    if (SMOKE) smokeLog.push({ level: 'warning', message: `[calls] signal ${data && data.type} ${data && data.candidate ? data.candidate.candidate : ''}` });
    if (!callWindow || callWindow.isDestroyed() || e.sender !== callWindow.webContents) return;
    sendToStudio('call:signal', { id: `${e.processId}:${e.frameId}`, data });
  });
  ipcMain.handle('call:signal-to', (_e, id, data) => {
    if (!callWindow || callWindow.isDestroyed() || typeof id !== 'string') return false;
    const [processId, frameId] = id.split(':').map(Number);
    callWindow.webContents.sendToFrame([processId, frameId], 'call:signal', data);
    return true;
  });
  if (SMOKE) {
    ipcMain.handle('smoke:call-result', async () => {
      if (!callWindow || callWindow.isDestroyed()) return null;
      return callWindow.webContents.executeJavaScript('window.__testResult || null').catch(() => null);
    });
  }
  ipcMain.handle('app:info', () => ({ version: app.getVersion(), name: 'Brainbox VirtualCam', packaged: app.isPackaged }));
  ipcMain.handle('updates:check', () => (updater ? updater.check() : null));
  ipcMain.handle('updates:install', () => {
    if (updater && updater.state.status === 'ready') updater.install();
    return true;
  });

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

  // Bundled files the studio needs as raw bytes (the Lite engine's runtime and models).
  const ASSETS = new Set(['mediapipe/vision_wasm_internal.wasm', 'mediapipe/face_landmarker.task', 'mediapipe/selfie_multiclass_256x256.tflite']);
  ipcMain.handle('asset:read', async (_e, name) => {
    if (!ASSETS.has(name)) throw new Error('Unknown asset');
    const data = await fs.promises.readFile(path.join(DIST, name));
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  });

  // ---------- local face engine ----------
  ipcMain.handle('faceengine:start', () => {
    if (!mainWindow) return { ok: false };
    if (!faceWorker) {
      const logFile = engineLogFile();
      faceWorker = utilityProcess.fork(path.join(__dirname, 'faceswap', 'worker.js'), [], {
        serviceName: 'Brainbox VirtualCam Face Engine',
        stdio: 'pipe',
        env: { ...process.env, VCAM_ENGINE_LOG: logFile },
      });
      const worker = faceWorker;
      // Anything the engine or the AI runtime prints (including native errors) goes to the log too.
      const append = (chunk) => fs.promises.appendFile(logFile, chunk).catch(() => {});
      if (worker.stdout) worker.stdout.on('data', append);
      if (worker.stderr) worker.stderr.on('data', append);
      worker.on('exit', (code) => {
        append(`[${new Date().toISOString()}] face engine exited with code ${code}\n`);
        if (faceWorker === worker) faceWorker = null;
        sendToStudio('faceengine:exit', { code });
      });
    }
    // A fresh channel each time the studio asks, so a reloaded window can reconnect.
    const { port1, port2 } = new MessageChannelMain();
    faceWorker.postMessage({ type: 'port' }, [port1]);
    mainWindow.webContents.postMessage('faceengine:port', null, [port2]);
    return { ok: true, modelsDir: modelsDir() };
  });

  // ---------- AI voice engine ----------
  // Its own process (so voice and face work never hold each other up). Two ports go to the studio:
  // one for control messages and one the audio worklet uses directly, so audio never waits on the page.
  ipcMain.handle('voiceengine:start', () => {
    if (!mainWindow) return { ok: false };
    if (!voiceWorker) {
      const logFile = engineLogFile();
      voiceWorker = utilityProcess.fork(path.join(__dirname, 'voice', 'worker.js'), [], {
        serviceName: 'Brainbox VirtualCam Voice Engine',
        stdio: 'pipe',
        env: { ...process.env, VCAM_ENGINE_LOG: logFile },
      });
      const worker = voiceWorker;
      const append = (chunk) => fs.promises.appendFile(logFile, chunk).catch(() => {});
      if (worker.stdout) worker.stdout.on('data', append);
      if (worker.stderr) worker.stderr.on('data', append);
      worker.on('exit', (code) => {
        append(`[${new Date().toISOString()}] voice engine exited with code ${code}\n`);
        if (voiceWorker === worker) voiceWorker = null;
        sendToStudio('voiceengine:exit', { code });
      });
    }
    const control = new MessageChannelMain();
    const audio = new MessageChannelMain();
    voiceWorker.postMessage({ type: 'ports' }, [control.port1, audio.port1]);
    mainWindow.webContents.postMessage('voiceengine:ports', null, [control.port2, audio.port2]);
    return { ok: true, modelsDir: modelsDir() };
  });

  ipcMain.handle('voiceengine:stop', () => {
    if (voiceWorker) voiceWorker.kill();
    voiceWorker = null;
    return true;
  });

  ipcMain.handle('faceengine:log', () => {
    const file = engineLogFile();
    if (fs.existsSync(file)) shell.showItemInFolder(file);
    else shell.openPath(path.dirname(file));
    return file;
  });

  ipcMain.handle('faceengine:stop', () => {
    if (faceWorker) faceWorker.kill();
    faceWorker = null;
    return true;
  });

  ipcMain.handle('hardware:check', async () => {
    if (!hardwareCache) {
      hardwareCache = await checkHardware(app);
      fs.promises
        .appendFile(engineLogFile(), `[${new Date().toISOString()}] hardware check: ${JSON.stringify(hardwareCache)}\n`)
        .catch(() => {});
    }
    return hardwareCache;
  });

  ipcMain.handle('models:status', () => ({ ...modelStatus(modelsDir()), busy: Boolean(downloader && downloader.busy) }));

  ipcMain.handle('models:download', async (_e, pack) => {
    if (pack !== 'core' && pack !== 'boost' && pack !== 'fast' && pack !== 'voices') throw new Error('Unknown model pack');
    if (!downloader) downloader = new Downloader(modelsDir());
    let lastSent = 0;
    return downloader.download(pack, (p) => {
      const now = Date.now();
      if (now - lastSent > 200 || p.done === p.total) {
        lastSent = now;
        sendToStudio('models:progress', { pack, ...p });
      }
    });
  });

  ipcMain.handle('models:cancel', () => {
    if (downloader) downloader.cancel();
    return true;
  });

  // ---------- VirtualCam camera device (Windows) ----------
  ipcMain.handle('camera:status', () => camera.refresh());
  ipcMain.handle('camera:install', () => camera.setInstalled(true));
  ipcMain.handle('camera:uninstall', () => camera.setInstalled(false));
  ipcMain.handle('camera:start', (_e, width, height) => camera.start(Number(width), Number(height)));
  ipcMain.handle('camera:stop', async () => {
    await camera.stop();
    return true;
  });
  ipcMain.on('camera:frame', (_e, frame) => {
    if (frame && frame.byteLength) camera.sendFrame(frame);
  });

  ipcMain.handle('smoke:save', async (_e, filePath, bytes) => {
    if (!SMOKE || typeof filePath !== 'string' || ![process.env.VCAM_SMOKE_LITE_OUT, process.env.VCAM_SMOKE_LOCAL_OUT, process.env.VCAM_SMOKE_LOOK_OUT].filter(Boolean).includes(filePath)) return false;
    await fs.promises.writeFile(filePath, Buffer.from(bytes));
    return true;
  });

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

// Self-test: install the tiny stand-in voice models (testdata/voice) as if the AI voices were
// downloaded, so the whole AI voice path can be checked without the real models.
function installSmokeVoices() {
  const src = process.env.VCAM_SMOKE_VOICE_DIR;
  if (!SMOKE || !src) return;
  const crypto = require('node:crypto');
  const dir = modelsDir();
  fs.mkdirSync(dir, { recursive: true });
  const files = ['contentvec.onnx', 'rmvpe.onnx', 'voice-test.onnx'].map((file) => {
    const data = fs.readFileSync(path.join(src, file));
    fs.writeFileSync(path.join(dir, file), data);
    return { file, label: file, url: `https://github.com/smoke/${file}`, sha256: crypto.createHash('sha256').update(data).digest('hex'), bytes: data.length };
  });
  const voice = { file: 'voice-test.onnx', sr: 40000, f0: 200 };
  const manifest = { version: 0, files, voices: { 'young-man': voice, 'young-woman': voice, 'older-man': voice, 'older-woman': voice } };
  fs.writeFileSync(path.join(dir, 'voice-models.json'), JSON.stringify(manifest));
}

app.whenReady().then(() => {
  installSmokeVoices();
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

  let cameraInUse = false;
  camera = new CameraDriver({
    binDir: vendorDir(),
    onState: (state) => {
      sendToStudio('camera:state', state);
      // A video app just opened the Brainbox VirtualCam camera.
      if (state.inUse && !cameraInUse) sendToStudio('camera-use', { kind: 'ours' });
      cameraInUse = Boolean(state.inUse);
    },
  });
  // Another app turning on a camera (usually your real webcam): the studio may ask to switch.
  cameraWatch = new CameraWatch({ onStart: ({ name }) => sendToStudio('camera-use', { kind: 'other', app: name }) });
  if (!SMOKE) cameraWatch.start();
  createTray();
  updater = setupUpdater((state) => sendToStudio('updates:state', state));
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
    }, 420000);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createMainWindow();
  });
});

app.on('before-quit', () => {
  app.isQuitting = true;
});

app.on('will-quit', () => {
  if (cameraWatch) cameraWatch.stop();
  globalShortcut.unregisterAll();
  if (camera) camera.stop();
  if (faceWorker) faceWorker.kill();
  if (voiceWorker) voiceWorker.kill();
  if (downloader) downloader.cancel();
});

app.on('window-all-closed', () => {
  // With the tray on, the app keeps running; the prompt card closing must never quit it either.
  if (tray && prefs.runInBackground && !app.isQuitting) return;
  app.quit();
});
