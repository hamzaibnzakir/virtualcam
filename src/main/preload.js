'use strict';

const { contextBridge, ipcRenderer, webUtils } = require('electron');

function listen(channel, handler) {
  const wrapped = (_event, payload) => handler(payload);
  ipcRenderer.on(channel, wrapped);
  return () => ipcRenderer.removeListener(channel, wrapped);
}

// The local face engine runs in its own process; the studio talks to it over a MessagePort.
let enginePort = null;
const engineListeners = new Set();
ipcRenderer.on('faceengine:port', (event) => {
  if (enginePort) enginePort.close();
  enginePort = event.ports[0];
  enginePort.onmessage = (e) => engineListeners.forEach((fn) => fn(e.data));
  enginePort.start();
});

// The AI voice engine's two ports go straight to the page (MessagePorts can't cross the bridge any
// other way). The page picks them up with window 'message' events of type 'brainbox-voice-ports'.
ipcRenderer.on('voiceengine:ports', (event) => {
  window.postMessage({ type: 'brainbox-voice-ports' }, '*', event.ports);
});

contextBridge.exposeInMainWorld('vcam', {
  platform: process.platform,
  smoke: process.env.VCAM_SMOKE === '1',
  smokeFiles:
    process.env.VCAM_SMOKE === '1'
      ? {
          face: process.env.VCAM_SMOKE_FACE || '',
          ref: process.env.VCAM_SMOKE_REF || '',
          installCamera: process.env.VCAM_SMOKE_INSTALL_CAMERA === '1',
          downloadModels: process.env.VCAM_SMOKE_DOWNLOAD === '1',
          liteOut: process.env.VCAM_SMOKE_LITE_OUT || '',
          localOut: process.env.VCAM_SMOKE_LOCAL_OUT || '',
          lookOut: process.env.VCAM_SMOKE_LOOK_OUT || '',
          voices: Boolean(process.env.VCAM_SMOKE_VOICE_DIR),
          tab: process.env.VCAM_SMOKE_TAB || '',
        }
      : null,

  loadState: () => ipcRenderer.invoke('state:load'),
  saveState: (state) => ipcRenderer.invoke('state:save', state),

  getSecret: (name) => ipcRenderer.invoke('secret:get', name),
  setSecret: (name, value) => ipcRenderer.invoke('secret:set', name, value),

  pickFile: (kind) => ipcRenderer.invoke('file:pick', kind),
  readFile: (filePath) => ipcRenderer.invoke('file:read', filePath),
  readAsset: (name) => ipcRenderer.invoke('asset:read', name),
  // Full path of a file dropped onto the window ('' when it isn't a file on disk).
  pathForFile: (file) => {
    try {
      return webUtils.getPathForFile(file) || '';
    } catch {
      return '';
    }
  },

  hotkeyFailures: () => ipcRenderer.invoke('hotkeys:failures'),
  onHotkey: (handler) => listen('hotkey', handler),

  publishScenes: (snapshot) => ipcRenderer.send('scenes:publish', snapshot),

  // Local face engine
  engineStart: () => ipcRenderer.invoke('faceengine:start'),
  engineStop: () => ipcRenderer.invoke('faceengine:stop'),
  enginePost: (message) => {
    if (!enginePort) return false;
    enginePort.postMessage(message);
    return true;
  },
  onEngineMessage: (handler) => {
    engineListeners.add(handler);
    return () => engineListeners.delete(handler);
  },
  onEngineExit: (handler) => listen('faceengine:exit', handler),
  showEngineLog: () => ipcRenderer.invoke('faceengine:log'),
  voiceEngineStart: () => ipcRenderer.invoke('voiceengine:start'),
  voiceEngineStop: () => ipcRenderer.invoke('voiceengine:stop'),
  onVoiceEngineExit: (handler) => listen('voiceengine:exit', handler),
  deleteMyVoice: (id) => ipcRenderer.invoke('voices:delete', id),
  saveVoiceMessage: (bytes, name) => ipcRenderer.invoke('voicemsg:save', bytes, name),

  // App preferences, the corner question card, camera use by other apps, updates.
  applyPrefs: (prefs) => ipcRenderer.invoke('prefs:apply', prefs),
  onPrefsSet: (handler) => listen('prefs:set', handler),
  showPrompt: (payload) => ipcRenderer.invoke('prompt:show', payload),
  onPromptAction: (handler) => listen('prompt:action', handler),
  onPromptData: (handler) => listen('prompt:data', handler),
  promptAction: (id) => ipcRenderer.invoke('prompt:action', id),
  onCameraUse: (handler) => listen('camera-use', handler),
  appInfo: () => ipcRenderer.invoke('app:info'),
  checkUpdates: () => ipcRenderer.invoke('updates:check'),
  installUpdate: () => ipcRenderer.invoke('updates:install'),
  onUpdates: (handler) => listen('updates:state', handler),

  // Calls window: open a web call app; the WebRTC link that carries the picture and voice into it.
  openCall: (target) => ipcRenderer.invoke('calls:open', target),
  onCallSignal: (handler) => listen('call:signal', handler),
  callSignal: (id, data) => ipcRenderer.invoke('call:signal-to', id, data),
  onCallClosed: (handler) => listen('call:closed', handler),
  smokeCallResult: () => ipcRenderer.invoke('smoke:call-result'),

  hardwareCheck: () => ipcRenderer.invoke('hardware:check'),
  modelsStatus: () => ipcRenderer.invoke('models:status'),
  modelsDownload: (pack) => ipcRenderer.invoke('models:download', pack),
  modelsCancel: () => ipcRenderer.invoke('models:cancel'),
  onModelsProgress: (handler) => listen('models:progress', handler),

  // VirtualCam camera device (Windows)
  cameraStatus: () => ipcRenderer.invoke('camera:status'),
  cameraInstall: () => ipcRenderer.invoke('camera:install'),
  cameraUninstall: () => ipcRenderer.invoke('camera:uninstall'),
  cameraStart: (width, height) => ipcRenderer.invoke('camera:start', width, height),
  cameraStop: () => ipcRenderer.invoke('camera:stop'),
  cameraFrame: (bgr) => ipcRenderer.send('camera:frame', bgr),
  onCameraState: (handler) => listen('camera:state', handler),

  // Quick switcher window
  onSwitcherScenes: (handler) => listen('switcher:scenes', handler),
  chooseScene: (index) => ipcRenderer.send('switcher:choose', index),
  panic: () => ipcRenderer.send('switcher:panic'),
  closeSwitcher: () => ipcRenderer.send('switcher:close'),

  smokeReport: (report) => ipcRenderer.send('smoke:report', report),
  smokeSave: (filePath, bytes) => ipcRenderer.invoke('smoke:save', filePath, bytes),
});
