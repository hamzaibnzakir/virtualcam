// Studio window: scenes, live preview, AI controls and the output window.
import { DecartAI } from './engine/ai-decart.js';
import { Compositor } from './engine/compositor.js';
import { SourceManager } from './engine/sources.js';
import {
  OUTPUT_FPS,
  PRESETS,
  fileName,
  makeScene,
  normalizeState,
  sceneKind,
  sourceLabel,
} from './engine/state.js';

const api = window.vcam;
const $ = (id) => document.getElementById(id);
const MOD = api.platform === 'darwin' ? 'Cmd Option' : 'Ctrl Alt';

let state = null;
let override = false; // panic mode: plain webcam, no AI, no overlays
let saveTimer = null;
let applyToken = 0;
let aiState = 'off';
let aiSeconds = 0;
let outputWindow = null;
let toastTimer = null;
let renderedSceneId = null; // scene whose values the text fields currently show
const refCache = new Map(); // reference image path -> Blob

// ---------- engine ----------

const sources = new SourceManager($('stage'), {
  readFile: (filePath) => api.readFile(filePath),
  onStatus: () => renderStatus(),
});
const compositor = new Compositor($('program'), $('ai-input'), sources);
const aiVideo = $('ai-output');
compositor.setAiVideo(aiVideo);

const ai = new DecartAI({
  getApiKey: () => api.getSecret('decartApiKey'),
  getInputStream: (mode) => compositor.aiInput(DecartAI.inputSizeFor(mode)),
  onRemoteStream: (stream) => {
    aiVideo.srcObject = stream || null;
    if (stream) aiVideo.play().catch(() => {});
  },
  onState: (next) => {
    aiState = next;
    compositor.aiLive = next === 'generating';
    renderAi();
  },
  onError: (message) => showToast(message, true),
});

// ---------- state helpers ----------

function activeScene() {
  return state.scenes.find((s) => s.id === state.activeId) || state.scenes[0];
}

function realCameraScene() {
  const cam = state.scenes.find((s) => s.source.type === 'webcam');
  return makeScene({ name: 'Real camera', source: { type: 'webcam', deviceId: cam ? cam.source.deviceId : '' } });
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    api.saveState(state).catch(() => showToast('Could not save your scenes', true));
  }, 400);
}

function publishScenes() {
  api.publishScenes({
    scenes: state.scenes.map((s, index) => ({ id: s.id, name: s.name, kind: sceneKind(s), index })),
    activeId: override ? null : state.activeId,
    override,
    mod: MOD,
  });
}

function changed({ reapply = true } = {}) {
  scheduleSave();
  publishScenes();
  if (reapply) applyScene();
  else renderAll();
}

async function loadReference(filePath) {
  if (refCache.has(filePath)) return refCache.get(filePath);
  const bytes = await api.readFile(filePath);
  const ext = filePath.split('.').pop().toLowerCase();
  const type = { png: 'image/png', webp: 'image/webp', gif: 'image/gif' }[ext] || 'image/jpeg';
  const blob = new Blob([bytes], { type });
  refCache.set(filePath, blob);
  return blob;
}

// ---------- applying a scene to the engine ----------

async function applyScene() {
  const token = ++applyToken;
  const scene = override ? realCameraScene() : activeScene();

  compositor.configure({ mirror: scene.mirror, nameTag: scene.nameTag });
  sources.use(scene.source);

  const wantAi = scene.ai.enabled;
  compositor.aiFeed = wantAi;
  renderAll();

  if (!wantAi) {
    ai.setTarget(null);
    return;
  }

  let refBlob = null;
  let refKey = '';
  if (scene.ai.referencePath) {
    try {
      refBlob = await loadReference(scene.ai.referencePath);
      refKey = scene.ai.referencePath;
    } catch {
      showToast('Reference image could not be loaded. Choose it again.', true);
    }
  }
  if (token !== applyToken) return; // user moved on while the image loaded

  ai.setTarget({
    mode: scene.ai.mode,
    prompt: scene.ai.prompt,
    enhance: scene.ai.enhance,
    refKey,
    refBlob,
  });
}

function selectScene(id) {
  if (!state.scenes.some((s) => s.id === id)) return;
  state.activeId = id;
  override = false;
  changed();
}

function selectSceneIndex(index) {
  const scene = state.scenes[index];
  if (scene) selectScene(scene.id);
}

function panic() {
  override = true;
  publishScenes();
  applyScene();
}

// ---------- rendering ----------

function setPressed(el, on) {
  el.setAttribute('aria-pressed', on ? 'true' : 'false');
}

const svg = (body) =>
  `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${body}</svg>`;
const THUMB_ICONS = {
  webcam: svg('<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3z"/>'),
  image: svg('<rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="9" cy="9" r="2"/><path d="M21 15l-5-5L5 21"/>'),
  video: svg('<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M10 9l5 3-5 3z"/>'),
  screen: svg('<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>'),
  card: svg('<path d="M5 6h14M12 6v13"/>'),
  ai: svg('<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M19 17l.7 2 .3.3 2 .7-2 .7-.3.3-.7 2-.7-2-.3-.3-2-.7 2-.7.3-.3z"/>'),
};

function renderScenes() {
  const list = $('scene-list');
  list.textContent = '';
  state.scenes.forEach((scene, index) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'scene';
    setPressed(btn, !override && scene.id === state.activeId);
    btn.addEventListener('click', () => selectScene(scene.id));

    const thumb = document.createElement('span');
    thumb.className = `scene-thumb${scene.ai.enabled ? ' ai' : ''}${scene.source.type === 'card' ? ' card' : ''}`;
    thumb.setAttribute('aria-hidden', 'true');
    thumb.innerHTML = THUMB_ICONS[scene.ai.enabled ? 'ai' : scene.source.type] || THUMB_ICONS.webcam;

    const meta = document.createElement('span');
    meta.className = 'scene-meta';
    const name = document.createElement('span');
    name.className = 'scene-name';
    name.textContent = scene.name;
    const kind = document.createElement('span');
    kind.className = 'scene-kind';
    kind.textContent = sceneKind(scene);
    meta.append(name, kind);

    btn.append(thumb, meta);
    if (index < 9) {
      const kbd = document.createElement('kbd');
      kbd.textContent = String(index + 1);
      kbd.title = `${MOD} ${index + 1}`;
      btn.append(kbd);
    }
    list.append(btn);
  });
}

// Text fields are not overwritten while you type in them, unless another scene was picked.
function syncField(input, value, sceneChanged) {
  if (sceneChanged || document.activeElement !== input) input.value = value;
}

function renderInspector() {
  const scene = activeScene();
  const index = state.scenes.indexOf(scene);
  const sceneChanged = renderedSceneId !== scene.id;
  renderedSceneId = scene.id;

  syncField($('scene-name'), scene.name, sceneChanged);
  $('scene-hotkey').textContent = index < 9 ? `Hotkey: ${MOD} ${index + 1}` : 'No hotkey (only the first 9 scenes get one)';

  document.querySelectorAll('#source-types button').forEach((b) => setPressed(b, b.dataset.source === scene.source.type));
  const type = scene.source.type;
  $('detail-webcam').hidden = type !== 'webcam';
  $('detail-file').hidden = type !== 'image' && type !== 'video';
  $('detail-screen').hidden = type !== 'screen';
  $('detail-card').hidden = type !== 'card';
  $('file-label').textContent = type === 'video' ? 'Video file (loops)' : 'Image file';
  $('file-name').textContent = fileName(scene.source.filePath) || 'No file chosen';
  $('camera-select').value = scene.source.deviceId;
  if ($('camera-select').value !== scene.source.deviceId) $('camera-select').value = '';
  syncField($('card-text'), scene.source.text, sceneChanged);
  $('card-color').value = scene.source.color;
  setPressed($('toggle-mirror'), scene.mirror);

  setPressed($('toggle-ai'), scene.ai.enabled);
  $('ai-mode').value = scene.ai.mode;
  setPressed($('toggle-enhance'), scene.ai.enhance);

  setPressed($('toggle-nametag'), scene.nameTag.enabled);
  syncField($('tag-name'), scene.nameTag.name, sceneChanged);
  syncField($('tag-title'), scene.nameTag.title, sceneChanged);
  syncField($('prompt'), scene.ai.prompt, sceneChanged);

  $('btn-delete-scene').disabled = state.scenes.length <= 1;
}

function renderPreviewMeta() {
  const scene = activeScene();
  $('scene-badge').textContent = override ? 'Real camera' : scene.name;
  $('override-badge').hidden = !override;
  $('pipe-source').textContent = override ? 'Webcam' : sourceLabel(scene.source.type);

  const presets = $('presets');
  presets.textContent = '';
  PRESETS.forEach((preset) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'preset';
    btn.textContent = preset.label;
    setPressed(btn, scene.ai.enabled && scene.ai.prompt === preset.prompt);
    btn.addEventListener('click', () => {
      const s = activeScene();
      s.ai.prompt = preset.prompt;
      s.ai.enabled = true;
      changed();
    });
    presets.append(btn);
  });

  const ref = scene.ai.referencePath;
  $('ref-title').textContent = ref ? fileName(ref) : 'Add a reference image';
  $('ref-sub').textContent = ref
    ? 'The AI uses this person or character as the look (Edit mode)'
    : 'Optional. The AI turns you into the person or character in the photo';
  $('btn-ref-clear').hidden = !ref;
}

const AI_TEXT = {
  off: ['AI is off for this scene', ''],
  'needs-key': ['Add your Decart API key in Settings', 'warn'],
  'needs-prompt': ['Type a look or pick a quick style', 'warn'],
  connecting: ['Connecting to Decart…', 'warn'],
  connected: ['Connected, starting the stream…', 'warn'],
  generating: ['AI live', 'on'],
  reconnecting: ['Connection dropped, reconnecting…', 'warn'],
  disconnected: ['AI disconnected, showing the real picture', 'err'],
  ended: ['AI session ended. Toggle AI to restart', 'err'],
  error: ['AI could not start. Check your key and connection', 'err'],
};

function renderAi() {
  const scene = activeScene();
  const showing = override || !scene.ai.enabled ? 'off' : aiState;
  const [text, tone] = AI_TEXT[showing] || [showing, ''];
  const status = $('ai-status');
  status.textContent = text;
  status.className = `ai-status${tone ? ` ${tone}` : ''}`;

  const chip = $('pipe-ai');
  const chipText = {
    off: 'AI off',
    generating: 'AI · Decart',
    'needs-key': 'AI needs key',
    'needs-prompt': 'AI needs a look',
    connecting: 'AI starting',
    connected: 'AI starting',
    reconnecting: 'AI reconnecting',
  };
  chip.textContent = chipText[showing] || 'AI stopped';
  chip.className = `chip${showing === 'generating' ? ' on' : showing === 'off' ? '' : ' warn'}`;
  $('foot-ai').textContent = showing === 'generating' ? 'AI live' : showing === 'off' ? 'AI off' : text;
}

function renderStatus() {
  const dot = $('status-dot');
  const text = $('status-text');
  if (override) {
    dot.className = 'dot off';
    text.textContent = 'Real camera (panic mode)';
  } else if (!sources.status.ready) {
    dot.className = 'dot warn';
    text.textContent = sources.status.message || 'Loading…';
  } else {
    dot.className = 'dot on';
    text.textContent = outputWindow && !outputWindow.closed ? 'Output window open' : 'Output ready';
  }
}

function renderCost() {
  const minutes = Math.floor(aiSeconds / 60);
  const seconds = String(aiSeconds % 60).padStart(2, '0');
  const time = `${minutes}:${seconds}`;
  const price = state.settings.pricePerSecond;
  if (price > 0) {
    $('cost-amount').textContent = `$${(aiSeconds * price).toFixed(2)}`;
    $('cost-note').textContent = `${time} of AI · at $${price}/s`;
  } else {
    $('cost-amount').textContent = time;
    $('cost-note').textContent = 'AI time. Add a price per second in Settings to see cost';
  }
}

function renderAll() {
  renderScenes();
  renderInspector();
  renderPreviewMeta();
  renderAi();
  renderStatus();
  renderCost();
}

// ---------- toast ----------

function showToast(message, isError = false) {
  const el = $('toast');
  el.textContent = message;
  el.className = `toast${isError ? ' err' : ''}`;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    el.hidden = true;
  }, isError ? 6000 : 3000);
}

// ---------- output window ----------

function openOutput() {
  if (outputWindow && !outputWindow.closed) {
    outputWindow.focus();
    return;
  }
  const win = window.open('about:blank', 'vcam-output');
  if (!win) {
    showToast('The output window could not open', true);
    return;
  }
  outputWindow = win;
  const doc = win.document;
  doc.title = 'VirtualCam Output';
  const style = doc.createElement('style');
  style.textContent = 'html,body{margin:0;height:100%;background:#000;overflow:hidden}video{width:100%;height:100%;object-fit:contain;display:block}';
  doc.head.append(style);
  const video = doc.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.srcObject = $('program').captureStream(OUTPUT_FPS);
  doc.body.append(video);
  video.play().catch(() => {});
  win.addEventListener('beforeunload', () => {
    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    outputWindow = null;
    renderStatus();
  });
  renderStatus();
}

// ---------- settings ----------

async function openSettings() {
  const key = await api.getSecret('decartApiKey');
  $('set-key').value = '';
  $('set-key').placeholder = key ? 'Key saved. Paste a new one to replace it' : 'Paste your key';
  $('key-status').textContent = key ? 'A key is saved on this computer.' : 'No key saved yet.';
  $('set-price').value = state.settings.pricePerSecond > 0 ? String(state.settings.pricePerSecond) : '';
  $('settings').showModal();
}

async function saveSettings() {
  const newKey = $('set-key').value.trim();
  if (newKey) {
    const result = await api.setSecret('decartApiKey', newKey);
    showToast(result.encrypted ? 'API key saved (encrypted)' : 'API key saved');
  }
  const price = Number($('set-price').value);
  state.settings.pricePerSecond = Number.isFinite(price) && price > 0 ? price : 0;
  $('settings').close();
  // A new key may unblock AI for the current scene.
  changed();
}

// ---------- wiring ----------

function bindEvents() {
  $('btn-new-scene').addEventListener('click', () => {
    const scene = makeScene({ name: `Scene ${state.scenes.length + 1}` });
    state.scenes.push(scene);
    selectScene(scene.id);
    $('scene-name').focus();
    $('scene-name').select();
  });

  $('btn-delete-scene').addEventListener('click', () => {
    if (state.scenes.length <= 1) return;
    const scene = activeScene();
    if (!window.confirm(`Delete "${scene.name}"?`)) return;
    const index = state.scenes.indexOf(scene);
    state.scenes.splice(index, 1);
    state.activeId = state.scenes[Math.max(0, index - 1)].id;
    changed();
  });

  $('scene-name').addEventListener('input', (e) => {
    activeScene().name = e.target.value.slice(0, 60) || 'Scene';
    changed({ reapply: false });
  });

  document.querySelectorAll('#source-types button').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const scene = activeScene();
      const type = btn.dataset.source;
      if (scene.source.type === type) return;
      const previousType = scene.source.type;
      scene.source.type = type;
      if (type === 'image' || type === 'video') {
        // Files are per type: an image path is no good for a video scene.
        if (previousType !== type) scene.source.filePath = '';
        changed();
        await pickFile(type);
        return;
      }
      changed();
    });
  });

  $('camera-select').addEventListener('change', (e) => {
    activeScene().source.deviceId = e.target.value;
    changed();
  });

  $('btn-pick-file').addEventListener('click', () => pickFile(activeScene().source.type));

  $('card-text').addEventListener('input', (e) => {
    activeScene().source.text = e.target.value.slice(0, 120);
    changed();
  });
  $('card-color').addEventListener('input', (e) => {
    activeScene().source.color = e.target.value;
    changed();
  });

  $('toggle-mirror').addEventListener('click', () => {
    const scene = activeScene();
    scene.mirror = !scene.mirror;
    changed();
  });

  $('toggle-ai').addEventListener('click', () => {
    const scene = activeScene();
    scene.ai.enabled = !scene.ai.enabled;
    changed();
  });

  $('ai-mode').addEventListener('change', (e) => {
    activeScene().ai.mode = e.target.value === 'edit' ? 'edit' : 'restyle';
    changed();
  });

  $('toggle-enhance').addEventListener('click', () => {
    const scene = activeScene();
    scene.ai.enhance = !scene.ai.enhance;
    changed();
  });

  const applyPrompt = () => {
    const scene = activeScene();
    scene.ai.prompt = $('prompt').value.trim().slice(0, 1000);
    scene.ai.enabled = true;
    changed();
  };
  $('btn-apply').addEventListener('click', applyPrompt);
  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') applyPrompt();
  });

  $('btn-ref').addEventListener('click', async () => {
    const filePath = await api.pickFile('image');
    if (!filePath) return;
    const scene = activeScene();
    scene.ai.referencePath = filePath;
    scene.ai.enabled = true;
    if (scene.ai.mode !== 'edit') {
      scene.ai.mode = 'edit';
      showToast('Switched to Edit mode, which uses reference images');
    }
    refCache.delete(filePath);
    changed();
  });
  $('btn-ref-clear').addEventListener('click', () => {
    activeScene().ai.referencePath = '';
    changed();
  });

  $('toggle-nametag').addEventListener('click', () => {
    const scene = activeScene();
    scene.nameTag.enabled = !scene.nameTag.enabled;
    changed();
  });
  $('tag-name').addEventListener('input', (e) => {
    activeScene().nameTag.name = e.target.value.slice(0, 60);
    changed();
  });
  $('tag-title').addEventListener('input', (e) => {
    activeScene().nameTag.title = e.target.value.slice(0, 80);
    changed();
  });

  $('btn-output').addEventListener('click', openOutput);
  $('btn-panic').addEventListener('click', panic);
  $('btn-settings').addEventListener('click', openSettings);
  $('btn-settings-cancel').addEventListener('click', () => $('settings').close());
  $('settings-form').addEventListener('submit', (e) => {
    e.preventDefault();
    saveSettings();
  });

  // Number keys switch scenes while the studio is focused (not while typing).
  document.addEventListener('keydown', (e) => {
    const target = e.target;
    const typing = target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.isContentEditable);
    if (typing || e.ctrlKey || e.metaKey || e.altKey || $('settings').open) return;
    if (/^[1-9]$/.test(e.key)) selectSceneIndex(Number(e.key) - 1);
    if (e.key === '0') panic();
  });

  api.onHotkey((msg) => {
    if (!msg) return;
    if (msg.action === 'scene') selectSceneIndex(msg.index);
    if (msg.action === 'panic') panic();
  });

  navigator.mediaDevices.addEventListener('devicechange', () => fillCameras());
}

async function pickFile(type) {
  const filePath = await api.pickFile(type === 'video' ? 'video' : 'image');
  if (!filePath) return;
  const scene = activeScene();
  if (scene.source.type !== type) return;
  scene.source.filePath = filePath;
  changed();
}

async function fillCameras() {
  const select = $('camera-select');
  const cams = await sources.listCameras();
  select.textContent = '';
  const def = document.createElement('option');
  def.value = '';
  def.textContent = 'Default camera';
  select.append(def);
  cams.forEach((cam) => {
    const opt = document.createElement('option');
    opt.value = cam.id;
    opt.textContent = cam.label;
    select.append(opt);
  });
  renderInspector();
}

function labelHotkeys() {
  $('kbd-panic').textContent = `${MOD} 0`;
  $('scenes-hint').textContent = `${MOD} 1–9 in any app`;
  $('foot-switcher').textContent = `${MOD} Space opens the quick switcher`;
}

async function reportHotkeyFailures() {
  const failed = await api.hotkeyFailures();
  if (failed.length) {
    showToast(`Some hotkeys are taken by another app: ${failed.join(', ').replace(/CommandOrControl/g, MOD.split(' ')[0])}`, true);
  }
}

// ---------- start ----------

async function start() {
  state = normalizeState(await api.loadState());
  labelHotkeys();
  bindEvents();
  compositor.start();

  setInterval(() => {
    if (aiState === 'generating' && !override && activeScene().ai.enabled) aiSeconds += 1;
    renderCost();
    $('foot-fps').textContent = `Output ${compositor.fps} fps`;
  }, 1000);

  publishScenes();
  await applyScene();
  await fillCameras();
  scheduleSave();
  reportHotkeyFailures();

  if (api.smoke) runSmokeTest();
}

// Automated self-test used by `npm run smoke`: exercises every scene type and reports back.
async function runSmokeTest() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const report = { steps: [] };
  const probe = document.createElement('canvas');
  probe.width = 1;
  probe.height = 1;
  const probeCtx = probe.getContext('2d', { willReadFrequently: true });
  const sample = () => {
    probeCtx.drawImage($('program'), 640, 360, 1, 1, 0, 0, 1, 1);
    const px = probeCtx.getImageData(0, 0, 1, 1).data;
    return { fps: compositor.fps, ready: sources.status.ready, message: sources.status.message, pixel: [px[0], px[1], px[2]] };
  };
  try {
    await wait(2500);
    report.steps.push({ step: 'webcam', ...sample() });

    selectScene(state.scenes[1].id); // AI scene with no key -> must fall back to the real picture
    await wait(1500);
    report.steps.push({ step: 'ai-without-key', aiState, ...sample() });

    const card = state.scenes.find((s) => s.source.type === 'card');
    selectScene(card.id);
    await wait(800);
    report.steps.push({ step: 'card', ...sample() });

    panic();
    await wait(1200);
    report.steps.push({ step: 'panic', override, ...sample() });

    $('btn-new-scene').click();
    await wait(300);
    const added = activeScene();
    added.nameTag = { enabled: true, name: 'Test Name', title: 'Test Title' };
    changed();
    await wait(1000);
    report.steps.push({ step: 'new-scene-nametag', scenes: state.scenes.length, ...sample() });

    openOutput();
    await wait(1000);
    report.steps.push({ step: 'output-window', open: Boolean(outputWindow && !outputWindow.closed) });

    // A bad key must fail cleanly and keep the real picture on screen.
    await api.setSecret('decartApiKey', 'invalid-smoke-test-key');
    selectScene(state.scenes[1].id);
    await wait(12000);
    report.steps.push({ step: 'ai-bad-key', aiState, toast: $('toast').textContent, ...sample() });
    await api.setSecret('decartApiKey', '');

    report.ok = true;
  } catch (err) {
    report.ok = false;
    report.error = String((err && err.stack) || err);
  }
  api.smokeReport(report);
}

window.addEventListener('error', (e) => showToast(`Something went wrong: ${e.message}`, true));
window.addEventListener('unhandledrejection', (e) => {
  const reason = e.reason;
  showToast(`Something went wrong: ${(reason && reason.message) || reason}`, true);
});

start().catch((err) => showToast(`VirtualCam could not start: ${err.message}`, true));
