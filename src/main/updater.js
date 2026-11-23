'use strict';

// Automatic updates from the public releases repo (hamzaibnzakir/brainbox-virtualcam-releases,
// installers only). Checks shortly after start and every few hours, downloads in the background,
// and installs when you click Restart (or the next time the app quits).
const { app } = require('electron');

const CHECK_EVERY_MS = 4 * 60 * 60 * 1000;

function setupUpdater(send) {
  const state = { version: app.getVersion(), status: 'idle', available: null, progress: 0, error: '' };
  const emit = (patch) => {
    Object.assign(state, patch);
    send(state);
  };
  if (!app.isPackaged || process.env.VCAM_SMOKE) {
    emit({ status: 'dev' });
    return { state, check: async () => state, install: () => {} };
  }
  let autoUpdater;
  try {
    ({ autoUpdater } = require('electron-updater'));
  } catch (err) {
    emit({ status: 'error', error: `Updater unavailable: ${err.message}` });
    return { state, check: async () => state, install: () => {} };
  }
  autoUpdater.autoDownload = true;
  autoUpdater.autoInstallOnAppQuit = true;
  autoUpdater.on('checking-for-update', () => emit({ status: 'checking', error: '' }));
  autoUpdater.on('update-available', (info) => emit({ status: 'downloading', available: info.version, progress: 0 }));
  autoUpdater.on('update-not-available', () => emit({ status: 'latest', available: null }));
  autoUpdater.on('download-progress', (p) => emit({ status: 'downloading', progress: Math.round(p.percent || 0) }));
  autoUpdater.on('update-downloaded', (info) => emit({ status: 'ready', available: info.version, progress: 100 }));
  autoUpdater.on('error', (err) => {
    // No releases repo yet, or offline: stay quiet in the studio, keep the reason for Settings.
    emit({ status: 'error', error: String((err && err.message) || err).split('\n')[0].slice(0, 200) });
  });

  const check = async () => {
    try {
      await autoUpdater.checkForUpdates();
    } catch {
      // reported through 'error'
    }
    return state;
  };
  setTimeout(check, 15000);
  setInterval(check, CHECK_EVERY_MS);
  return {
    state,
    check,
    install: () => {
      app.isQuitting = true;
      autoUpdater.quitAndInstall(false, true);
    },
  };
}

module.exports = { setupUpdater };
