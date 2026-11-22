'use strict';

const { contextBridge, ipcRenderer } = require('electron');

function listen(channel, handler) {
  const wrapped = (_event, payload) => handler(payload);
  ipcRenderer.on(channel, wrapped);
  return () => ipcRenderer.removeListener(channel, wrapped);
}

contextBridge.exposeInMainWorld('vcam', {
  platform: process.platform,
  smoke: process.env.VCAM_SMOKE === '1',

  loadState: () => ipcRenderer.invoke('state:load'),
  saveState: (state) => ipcRenderer.invoke('state:save', state),

  getSecret: (name) => ipcRenderer.invoke('secret:get', name),
  setSecret: (name, value) => ipcRenderer.invoke('secret:set', name, value),

  pickFile: (kind) => ipcRenderer.invoke('file:pick', kind),
  readFile: (filePath) => ipcRenderer.invoke('file:read', filePath),

  hotkeyFailures: () => ipcRenderer.invoke('hotkeys:failures'),
  onHotkey: (handler) => listen('hotkey', handler),

  publishScenes: (snapshot) => ipcRenderer.send('scenes:publish', snapshot),

  // Quick switcher window
  onSwitcherScenes: (handler) => listen('switcher:scenes', handler),
  chooseScene: (index) => ipcRenderer.send('switcher:choose', index),
  panic: () => ipcRenderer.send('switcher:panic'),
  closeSwitcher: () => ipcRenderer.send('switcher:close'),

  smokeReport: (report) => ipcRenderer.send('smoke:report', report),
});
