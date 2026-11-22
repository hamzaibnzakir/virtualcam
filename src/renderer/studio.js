// Studio window: scenes, live preview, AI controls and the output window.
import { DecartAI } from './engine/ai-decart.js';
import { LiteFaceAI } from './engine/ai-lite.js';
import { LocalFaceAI } from './engine/ai-local.js';
import { Compositor } from './engine/compositor.js';
import { SourceManager } from './engine/sources.js';
import {
  CHARACTER_PROMPT,
  OUTPUT_FPS,
  output,
  setOrientation,
  PRESETS,
  isPhoneCamera,
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
let activeEngine = 'none'; // which AI the current scene uses: 'cloud' | 'local' | 'none'
let cloudState = 'off';
let hardware = null;
let models = { core: false, boost: false, coreBytes: 0, boostBytes: 0, busy: false };
let download = null; // { pack, done, total, file }
let cameraState = { supported: false, installed: false, running: false, inUse: false, message: '', width: 0, height: 0 };
const refCache = new Map(); // reference image path -> Blob

// ---------- engine ----------

const sources = new SourceManager($('stage'), {
  readFile: (filePath) => api.readFile(filePath),
  onStatus: () => renderStatus(),
});
const compositor = new Compositor($('program'), $('ai-input'), sources);
const aiVideo = $('ai-output');

const cloud = new DecartAI({
  getApiKey: () => api.getSecret('decartApiKey'),
  getInputStream: (mode) => compositor.aiInput(DecartAI.inputSizeFor(mode)),
  onRemoteStream: (stream) => {
    aiVideo.srcObject = stream || null;
    if (stream) aiVideo.play().catch(() => {});
  },
  onState: (next) => {
    cloudState = next;
    if (activeEngine === 'cloud') {
      aiState = next;
      renderAi();
    }
  },
  onError: (message) => showToast(message, true),
});

const local = new LocalFaceAI({
  api,
  onState: (next) => {
    if (activeEngine === 'local') {
      aiState = next;
      renderAi();
    }
    renderLocal();
  },
  onProgress: () => {
    if (activeEngine === 'local') renderAi();
  },
  onError: (message) => showToast(message, true),
});

const lite = new LiteFaceAI({
  api,
  onState: (next) => {
    if (activeEngine === 'lite') {
      aiState = next;
      renderAi();
    }
  },
  onError: (message) => showToast(message, true),
});

// The local engines grab frames as they are drawn; the camera driver takes each finished frame.
compositor.beforeRender = (drawSource) => {
  if (activeEngine === 'local') local.tick(drawSource);
  else if (activeEngine === 'lite') lite.tick(drawSource);
};
compositor.afterRender = (canvas) => sendCameraFrame(canvas);

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

  compositor.configure({ mirror: scene.mirror, nameTag: scene.nameTag, fit: scene.fit });
  sources.use(scene.source);

  const wantAi = scene.ai.enabled;
  const engine = wantAi ? scene.ai.engine : 'none';
  if (engine !== activeEngine) {
    activeEngine = engine;
    aiState = engine === 'cloud' ? cloudState : engine === 'local' ? local.state : engine === 'lite' ? lite.state : 'off';
  }
  compositor.aiFeed = engine === 'cloud';
  if (engine === 'cloud') compositor.setAiSource(aiVideo, () => cloudState === 'generating');
  else if (engine === 'local') compositor.setAiSource(local.canvas, () => local.live);
  else if (engine === 'lite') compositor.setAiSource(lite.canvas, () => lite.live);
  else compositor.setAiSource(null);
  renderAll();

  if (engine !== 'cloud') cloud.setTarget(null);
  if (engine !== 'local') local.setTarget(null);
  if (engine !== 'lite') lite.setTarget(null);
  if (engine === 'none') return;

  let refBlob = null;
  let refKey = '';
  if (scene.ai.referencePath) {
    try {
      refBlob = await loadReference(scene.ai.referencePath);
      refKey = scene.ai.referencePath;
    } catch {
      showToast('That photo could not be loaded. Choose it again.', true);
    }
  }
  if (token !== applyToken) return; // user moved on while the image loaded

  if (engine === 'local') {
    local.setTarget({ refKey, refBlob, enhance: state.settings.faceBoost });
    return;
  }
  if (engine === 'lite') {
    lite.setTarget({ refKey, refBlob });
    return;
  }

  // Cloud: a reference photo means character swap, which the edit model does.
  const prompt = scene.ai.prompt || (refBlob ? CHARACTER_PROMPT : '');
  cloud.setTarget({
    mode: refBlob ? 'edit' : scene.ai.mode,
    prompt,
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
  setPressed($('toggle-fit'), scene.fit === 'fit');

  setPressed($('toggle-ai'), scene.ai.enabled);
  $('ai-engine').value = scene.ai.engine;
  $('cloud-group').hidden = scene.ai.engine !== 'cloud';
  $('local-group').hidden = scene.ai.engine !== 'local';
  $('lite-group').hidden = scene.ai.engine !== 'lite';
  $('ai-mode').value = scene.ai.referencePath ? 'edit' : scene.ai.mode;
  $('ai-mode').disabled = Boolean(scene.ai.referencePath);
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
  // Prompts and styles only mean something to the cloud AI.
  const isLocal = scene.ai.engine === 'local' || scene.ai.engine === 'lite';
  $('look-field').hidden = isLocal;
  $('styles-field').hidden = isLocal;

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
  $('ref-title').textContent = ref ? `Becoming: ${fileName(ref)}` : 'Drop a photo or click to choose';
  $('ref-sub').textContent = ref
    ? scene.ai.engine === 'lite'
      ? 'You wear this face. Your eyes, mouth, hair and moves stay yours.'
      : isLocal
      ? 'Your face becomes this face. Your hair, clothes and moves stay yours.'
      : 'The cloud AI turns you into this person: face, hair and outfit, following your moves.'
    : 'The AI turns you into the person in the photo, following your moves';
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
  'needs-models': ['Download the local AI models below (one time)', 'warn'],
  starting: ['Starting the face engine…', 'warn'],
  loading: ['Reading the face in your photo…', 'warn'],
  'needs-reference': ['Drop a photo of the person you want to become', 'warn'],
  'no-face': ['No face found in that photo. Try a clear, front-facing one', 'err'],
};

function renderAi() {
  const scene = activeScene();
  const showing = override || !scene.ai.enabled ? 'off' : aiState;
  let [text, tone] = AI_TEXT[showing] || [showing, ''];
  if (activeEngine === 'local' && showing === 'generating') text = 'Local face swap live';
  if (activeEngine === 'local' && showing === 'error') text = 'The local face engine hit a problem';
  if (activeEngine === 'local' && (showing === 'starting' || showing === 'loading') && local.step) text = `Local AI: ${local.step}…`;
  if (activeEngine === 'lite' && showing === 'generating') text = lite.faceFound ? 'Lite face warp live' : 'Lite: looking for your face…';
  if (activeEngine === 'lite' && showing === 'starting') text = 'Starting face tracking…';
  if (activeEngine === 'lite' && showing === 'error') text = 'Lite face warp hit a problem';
  const status = $('ai-status');
  status.textContent = text;
  status.className = `ai-status${tone ? ` ${tone}` : ''}`;

  const chip = $('pipe-ai');
  const chipText = {
    off: 'AI off',
    generating: activeEngine === 'local' ? 'AI · local' : activeEngine === 'lite' ? 'AI · lite' : 'AI · Decart',
    starting: 'AI starting',
    loading: 'AI starting',
    'needs-models': 'AI needs models',
    'needs-reference': 'AI needs a photo',
    'needs-key': 'AI needs key',
    'needs-prompt': 'AI needs a look',
    connecting: 'AI starting',
    connected: 'AI starting',
    reconnecting: 'AI reconnecting',
  };
  chip.textContent = chipText[showing] || 'AI stopped';
  chip.className = `chip${showing === 'generating' ? ' on' : showing === 'off' ? '' : ' warn'}`;
  $('foot-ai').textContent = showing === 'generating' ? 'AI live' : showing === 'off' ? 'AI off' : text;
  renderLocal();
}

function formatMB(bytes) {
  return bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
}

function renderLocal() {
  if (!state) return;
  if (hardware) {
    $('hw-line').textContent = hardware.message || 'Ready.';
  }
  const busyCore = download && download.pack === 'core';
  const busyBoost = download && download.pack === 'boost';
  if (busyCore || busyBoost) {
    const pct = download.total ? Math.floor((download.done / download.total) * 100) : 0;
    $('models-progress').hidden = false;
    $('models-bar').style.width = `${pct}%`;
    $('models-line').textContent = `Downloading ${download.file}… ${pct}%`;
  } else {
    $('models-progress').hidden = true;
    $('models-line').textContent = models.core
      ? 'Local AI models ready'
      : `Local AI models needed (${formatMB(models.coreBytes)}, one time)`;
  }
  const btn = $('btn-models');
  btn.hidden = models.core && !busyCore;
  btn.textContent = busyCore ? 'Cancel' : 'Download';
  btn.disabled = Boolean(busyBoost);

  setPressed($('toggle-boost'), state.settings.faceBoost);
  $('boost-row').hidden = models.boost && !busyBoost;
  $('boost-line').textContent = busyBoost ? 'Downloading the detail model…' : `Needs a one-time download (${formatMB(models.boostBytes)})`;
  $('btn-boost-download').textContent = busyBoost ? 'Cancel' : 'Download';
  $('btn-boost-download').disabled = Boolean(busyCore);

  const liteLive = activeEngine === 'lite' && lite.state === 'generating';
  $('lite-stats').textContent = liteLive
    ? `${lite.stats.fps} frames per second · ${lite.stats.ms} ms each · tracking on the ${lite.delegate === 'GPU' ? 'graphics chip' : 'processor'}`
    : '';
  const live = activeEngine === 'local' && local.state === 'generating';
  const where = local.deviceLabel || (local.provider === 'dml' ? 'your graphics card' : local.provider === 'coreml' ? 'your Mac' : 'the processor');
  const usable = local.tried.filter((t) => !t.error).length;
  const st = local.stats.stages;
  const split = st && st.enhance !== undefined ? ` (swap ${st.swap} ms + boost ${st.enhance} ms)` : '';
  $('local-stats').textContent = live
    ? `${local.tracker ? 'Your face is tracked every frame; the AI refreshes it ' : ''}${local.stats.fps} ${local.tracker ? 'times' : 'swaps'} per second · ${local.stats.ms} ms each${split} · AI on ${where}${usable > 1 ? ` (the fastest of ${usable} tried)` : ''}`
    : '';
}

async function refreshModels() {
  try {
    models = await api.modelsStatus();
  } catch {
    // keep the last known status
  }
  renderLocal();
}

async function startDownload(pack) {
  if (download) {
    api.modelsCancel();
    return;
  }
  download = { pack, done: 0, total: 1, file: 'models' };
  renderLocal();
  try {
    models = await api.modelsDownload(pack);
    showToast(pack === 'core' ? 'Local AI models downloaded' : 'Face detail boost downloaded');
  } catch (err) {
    showToast((err && err.message) || 'Download failed', true);
  } finally {
    download = null;
    await refreshModels();
    applyScene();
  }
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
  if (activeEngine === 'local' || activeEngine === 'lite') {
    $('cost-amount').textContent = 'Free';
    $('cost-note').textContent = 'This engine runs on this computer';
    return;
  }
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

// ---------- VirtualCam camera device ----------

const camCanvas = document.createElement('canvas');
const camCtx = camCanvas.getContext('2d', { willReadFrequently: true, alpha: false });
let camFrame = new Uint8Array(0);

// Only spends time on frames while some app is actually showing the camera.
function sendCameraFrame(canvas) {
  if (!cameraState.running || !cameraState.inUse || !state || !state.settings.cameraOutput) return;
  if (canvas.width !== cameraState.width || canvas.height !== cameraState.height) return; // camera restarting
  if (camCanvas.width !== canvas.width || camCanvas.height !== canvas.height) {
    camCanvas.width = canvas.width;
    camCanvas.height = canvas.height;
  }
  if (camFrame.length !== canvas.width * canvas.height * 3) camFrame = new Uint8Array(canvas.width * canvas.height * 3);
  camCtx.drawImage(canvas, 0, 0);
  const rgba = camCtx.getImageData(0, 0, canvas.width, canvas.height).data;
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    camFrame[j] = rgba[i + 2];
    camFrame[j + 1] = rgba[i + 1];
    camFrame[j + 2] = rgba[i];
  }
  api.cameraFrame(camFrame);
}

function renderCamera() {
  const btn = $('btn-camera');
  btn.hidden = !cameraState.supported;
  btn.classList.toggle('on', cameraState.installed && cameraState.running);
  if (!cameraState.installed) btn.textContent = 'Install VirtualCam camera';
  else if (!state || !state.settings.cameraOutput) btn.textContent = 'Camera paused';
  else if (cameraState.inUse) btn.textContent = 'VirtualCam: in use';
  else if (cameraState.running) btn.textContent = 'VirtualCam: ready';
  else btn.textContent = 'VirtualCam: starting…';

  const detail = $('camera-detail');
  if (!cameraState.supported) detail.textContent = 'The camera device is Windows only for now. Use the output window with OBS.';
  else if (cameraState.message) detail.textContent = cameraState.message;
  else if (cameraState.installed) detail.textContent = 'Installed. Pick "VirtualCam" as the camera in Zoom, Meet, Teams, Discord or OBS.';
  else detail.textContent = 'Not installed. Installing needs admin approval once.';
  $('btn-camera-install').hidden = !cameraState.supported || cameraState.installed;
  $('btn-camera-remove').hidden = !cameraState.supported || !cameraState.installed;
  if (state) setPressed($('toggle-camout'), state.settings.cameraOutput);
}

async function syncCamera() {
  if (!cameraState.supported) return;
  if (cameraState.installed && state.settings.cameraOutput) await api.cameraStart(output.width, output.height);
  else await api.cameraStop();
}

async function installCamera() {
  showToast('Windows will ask for permission to add the camera…');
  cameraState = await api.cameraInstall();
  renderCamera();
  if (cameraState.installed) {
    showToast('VirtualCam camera installed. Pick it in your video app.');
    await syncCamera();
  }
}

// ---------- settings ----------

async function openSettings() {
  const key = await api.getSecret('decartApiKey');
  $('set-key').value = '';
  $('set-key').placeholder = key ? 'Key saved. Paste a new one to replace it' : 'Paste your key';
  $('key-status').textContent = key ? 'A key is saved on this computer.' : 'No key saved yet.';
  $('set-price').value = state.settings.pricePerSecond > 0 ? String(state.settings.pricePerSecond) : '';
  renderCamera();
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

  $('toggle-fit').addEventListener('click', () => {
    const scene = activeScene();
    scene.fit = scene.fit === 'fit' ? 'fill' : 'fit';
    changed();
  });

  $('output-shape').addEventListener('change', async (e) => {
    state.settings.orientation = e.target.value === 'portrait' ? 'portrait' : 'landscape';
    applyOrientation();
    scheduleSave();
    await syncCamera();
    if (cameraState.installed && state.settings.cameraOutput) {
      showToast('Picture shape changed. In an open call, pick VirtualCam again; in Chrome or Meet, reload the page.');
    }
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
    if (filePath) setReference(filePath);
  });

  // Drop a photo anywhere on the window to become that person.
  let dragDepth = 0;
  const hasFiles = (e) => Array.from((e.dataTransfer && e.dataTransfer.types) || []).includes('Files');
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth += 1;
    $('drop-overlay').hidden = false;
  });
  window.addEventListener('dragover', (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) $('drop-overlay').hidden = true;
  });
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    dragDepth = 0;
    $('drop-overlay').hidden = true;
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!file) return;
    if (!/^image\/(png|jpe?g|webp|gif)$/.test(file.type)) {
      showToast('Drop a photo (PNG, JPG or WebP)', true);
      return;
    }
    const filePath = api.pathForFile(file);
    if (!filePath) {
      showToast('That photo is not a file on this computer. Save it first, then drop it.', true);
      return;
    }
    setReference(filePath);
  });

  $('ai-engine').addEventListener('change', (e) => {
    const value = e.target.value;
    activeScene().ai.engine = value === 'local' || value === 'lite' ? value : 'cloud';
    changed();
  });
  $('btn-models').addEventListener('click', () => startDownload('core'));
  $('btn-engine-log').addEventListener('click', () => api.showEngineLog());
  $('btn-boost-download').addEventListener('click', () => startDownload('boost'));
  $('toggle-boost').addEventListener('click', () => {
    if (!state.settings.faceBoost && !models.boost) {
      startDownload('boost').then(() => {
        if (models.boost) {
          state.settings.faceBoost = true;
          changed();
        }
      });
      return;
    }
    state.settings.faceBoost = !state.settings.faceBoost;
    changed();
  });

  $('btn-phone-help').addEventListener('click', () => $('phone-help').showModal());
  $('btn-refresh-cams').addEventListener('click', () => fillCameras());

  $('btn-camera').addEventListener('click', () => {
    if (!cameraState.installed) installCamera();
    else openSettings();
  });
  $('btn-camera-install').addEventListener('click', installCamera);
  $('btn-camera-remove').addEventListener('click', async () => {
    if (!window.confirm('Remove the VirtualCam camera from Windows? Video apps will no longer see it.')) return;
    cameraState = await api.cameraUninstall();
    renderCamera();
  });
  $('toggle-camout').addEventListener('click', async () => {
    state.settings.cameraOutput = !state.settings.cameraOutput;
    scheduleSave();
    renderCamera();
    await syncCamera();
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

function setReference(filePath) {
  const scene = activeScene();
  scene.ai.referencePath = filePath;
  scene.ai.enabled = true;
  refCache.delete(filePath);
  changed();
  showToast(`Becoming the person in ${fileName(filePath)}`);
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
  cams
    // Never offer our own output as an input: that would loop the picture into itself.
    .filter((cam) => !/^virtualcam/i.test(cam.label))
    .forEach((cam) => {
      const opt = document.createElement('option');
      opt.value = cam.id;
      opt.textContent = isPhoneCamera(cam.label) ? `${cam.label} (phone)` : cam.label;
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

function applyOrientation() {
  setOrientation(state.settings.orientation);
  $('output-shape').value = output.orientation;
  $('program').closest('.preview').classList.toggle('portrait', output.orientation === 'portrait');
  // The cloud AI's own size never changes, so it keeps running across the switch.
}

async function start() {
  state = normalizeState(await api.loadState());
  applyOrientation();
  labelHotkeys();
  bindEvents();
  compositor.start();

  setInterval(() => {
    if (activeEngine === 'cloud' && cloudState === 'generating' && !override) aiSeconds += 1;
    renderCost();
    if (activeEngine === 'local' || activeEngine === 'lite') renderLocal();
    $('foot-fps').textContent = `Output ${compositor.fps} fps`;
  }, 1000);

  api.onModelsProgress((p) => {
    if (download && p.pack === download.pack) {
      download = { ...download, ...p };
      renderLocal();
    }
  });
  api.onCameraState((next) => {
    cameraState = next;
    renderCamera();
  });

  publishScenes();
  await applyScene();
  await fillCameras();
  scheduleSave();
  reportHotkeyFailures();

  api.hardwareCheck().then((info) => {
    hardware = info;
    local.gpuCount = ((info && info.gpus) || []).filter((g) => !/microsoft (basic|remote)/i.test(g.name)).length;
    renderLocal();
  });
  refreshModels();
  cameraState = await api.cameraStatus();
  renderCamera();
  await syncCamera();

  if (api.smoke) runSmokeTest();
}

// Automated self-test used by `npm run smoke`: exercises every scene type and reports back.
async function runSmokeTest() {
  const wait = (ms) => new Promise((r) => setTimeout(r, ms));
  const report = { steps: [] };
  const failures = [];
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

    report.steps.push({ step: 'camera-device', ...cameraState, buttonHidden: $('btn-camera').hidden });

    // A bad key must fail cleanly and keep the real picture on screen.
    await api.setSecret('decartApiKey', 'invalid-smoke-test-key');
    selectScene(state.scenes[1].id);
    await wait(12000);
    report.steps.push({ step: 'ai-bad-key', aiState, toast: $('toast').textContent, ...sample() });
    await api.setSecret('decartApiKey', '');

    // Windows: install the camera the way the Install button does, then watch it like a video app would.
    if (api.smokeFiles && api.smokeFiles.installCamera && cameraState.supported) {
      await installCamera();
      const card = state.scenes.find((s) => s.source.type === 'card');
      card.source.text = ' ';
      card.source.color = '#20c864';
      selectScene(card.id);
      await wait(1500);
      // Device names stay hidden until camera permission is granted, so ask for any camera first.
      try {
        const probe = await navigator.mediaDevices.getUserMedia({ video: true });
        probe.getTracks().forEach((t) => t.stop());
      } catch (err) {
        report.steps.push({ step: 'camera-permission', error: String((err && err.name) || err) });
      }
      const devices = await navigator.mediaDevices.enumerateDevices();
      const vc = devices.find((d) => d.kind === 'videoinput' && /virtualcam/i.test(d.label));
      const step = {
        step: 'camera-loopback',
        installed: cameraState.installed,
        running: cameraState.running,
        listed: Boolean(vc),
        cameras: devices.filter((d) => d.kind === 'videoinput').map((d) => d.label || '(no name)'),
      };
      if (vc) {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: vc.deviceId }, width: 1280, height: 720 } });
        const video = document.createElement('video');
        video.muted = true;
        video.srcObject = stream;
        $('stage').append(video);
        await video.play();
        await wait(3000);
        const c = document.createElement('canvas');
        c.width = video.videoWidth;
        c.height = video.videoHeight;
        const cx = c.getContext('2d', { willReadFrequently: true });
        cx.drawImage(video, 0, 0);
        const px = cx.getImageData(Math.floor(c.width / 2), Math.floor(c.height / 2), 1, 1).data;
        step.size = `${c.width}x${c.height}`;
        step.pixel = [px[0], px[1], px[2]];
        step.inUse = cameraState.inUse;
        stream.getTracks().forEach((t) => t.stop());
        video.remove();
        step.ok = Math.abs(px[0] - 32) < 14 && Math.abs(px[1] - 200) < 14 && Math.abs(px[2] - 100) < 14;
      }
      report.steps.push(step);
      if (!step.ok) failures.push('the VirtualCam camera did not show the studio picture');
    }

    // Lite face warp on a still photo of a face.
    if (api.smokeFiles && api.smokeFiles.face && api.smokeFiles.ref) {
      const scene = makeScene({
        name: 'Smoke lite warp',
        source: { type: 'image', filePath: api.smokeFiles.face },
        ai: { enabled: true, engine: 'lite', referencePath: api.smokeFiles.ref },
      });
      state.scenes.push(scene);
      selectScene(scene.id);
      const until = Date.now() + 60000;
      while (Date.now() < until && !(lite.state === 'generating' && lite.faceFound && lite.frame > 10)) await wait(250);
      const step = { step: 'lite-warp', aiState, delegate: lite.delegate, faceFound: lite.faceFound, frames: lite.frame, ms: lite.stats.ms, fps: lite.stats.fps };
      if (api.smokeFiles.liteOut) {
        const png = await new Promise((r) => lite.canvas.toBlob(r, 'image/png'));
        step.saved = await api.smokeSave(api.smokeFiles.liteOut, new Uint8Array(await png.arrayBuffer()));
      }
      report.steps.push(step);
      if (!(lite.faceFound && lite.frame > 10)) failures.push('lite face warp did not track the face');
    }

    // Local face swap on a still photo of a face, when the test models are available.
    if (api.smokeFiles && api.smokeFiles.face && api.smokeFiles.ref) {
      await refreshModels();
      if (!models.core && api.smokeFiles.downloadModels) {
        const t0 = Date.now();
        await startDownload('core');
        report.steps.push({ step: 'model-download', ok: models.core, seconds: Math.round((Date.now() - t0) / 1000) });
      }
      const scene = makeScene({
        name: 'Smoke local swap',
        source: { type: 'image', filePath: api.smokeFiles.face },
        ai: { enabled: true, engine: 'local', referencePath: api.smokeFiles.ref },
      });
      state.scenes.push(scene);
      selectScene(scene.id);
      const until = Date.now() + 90000;
      while (Date.now() < until && local.lastResultAt === 0) await wait(500);
      await wait(500);
      // How much the face area changed between what went in and what came out.
      const a = local.inputCtx.getImageData(600, 300, 80, 80).data;
      const b = local.ctx.getImageData(600, 300, 80, 80).data;
      let diff = 0;
      for (let i = 0; i < a.length; i += 4) diff += Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]);
      let localSaved = false;
      if (api.smokeFiles.localOut) {
        const png = await new Promise((r) => local.canvas.toBlob(r, 'image/png'));
        localSaved = await api.smokeSave(api.smokeFiles.localOut, new Uint8Array(await png.arrayBuffer()));
      }
      report.steps.push({
        step: 'local-swap',
        tracked: Boolean(local.tracker),
        faceTracked: Boolean(local.points),
        saved: localSaved,
        models: models.core,
        aiState,
        provider: local.provider,
        gpuFallback: local.fallbackReason,
        swapped: local.lastResultAt > 0,
        live: local.live,
        meanFaceChange: Math.round(diff / (80 * 80 * 3)),
        ms: local.stats.ms,
        hardware: hardware && hardware.verdict,
        gpus: hardware && hardware.gpus,
        gpuSources: hardware && hardware.sources,
        device: local.deviceLabel,
        tried: local.tried,
      });
      if (!(local.lastResultAt > 0)) failures.push('local face swap produced no result');
    }


    // Portrait output: every engine and the camera follow the new shape; 'fit' shows the whole picture.
    if (api.smokeFiles && api.smokeFiles.face && api.smokeFiles.ref) {
      const select = $('output-shape');
      select.value = 'portrait';
      select.dispatchEvent(new Event('change'));
      const portraitStep = { step: 'portrait' };
      const liteScene = state.scenes.find((s) => s.name === 'Smoke lite warp');
      liteScene.fit = 'fit';
      selectScene(liteScene.id);
      const startFrame = lite.frame;
      const until = Date.now() + 60000;
      while (Date.now() < until && lite.frame < startFrame + 5) await wait(250);
      const program = $('program');
      portraitStep.program = `${program.width}x${program.height}`;
      portraitStep.lite = `${lite.canvas.width}x${lite.canvas.height}`;
      portraitStep.liteFace = lite.faceFound;
      // Top-left corner should be the soft blurred fill, not the empty-source placeholder.
      probeCtx.drawImage(program, 4, 4, 1, 1, 0, 0, 1, 1);
      const corner = probeCtx.getImageData(0, 0, 1, 1).data;
      portraitStep.corner = [corner[0], corner[1], corner[2]];
      portraitStep.ok = program.width === 720 && program.height === 1280 && lite.canvas.width === 720 && lite.faceFound;
      if (cameraState.supported && cameraState.installed) {
        const until2 = Date.now() + 15000;
        while (Date.now() < until2 && !(cameraState.running && cameraState.width === 720)) await wait(250);
        // What matters: the camera itself now sends 720x1280 (the driver test captures it fresh).
        portraitStep.bridge = `${cameraState.width}x${cameraState.height}`;
        portraitStep.ok = portraitStep.ok && cameraState.running && cameraState.width === 720 && cameraState.height === 1280;
        const devices = await navigator.mediaDevices.enumerateDevices();
        const vc = devices.find((d) => d.kind === 'videoinput' && /virtualcam/i.test(d.label));
        if (vc) {
          const stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: vc.deviceId } } });
          const video = document.createElement('video');
          video.muted = true;
          video.srcObject = stream;
          $('stage').append(video);
          await video.play();
          await wait(2500);
          // Informational: Chromium keeps the sizes it learned at startup until a device change.
          portraitStep.chromiumSees = `${video.videoWidth}x${video.videoHeight}`;
          stream.getTracks().forEach((t) => t.stop());
          video.remove();
        }
      }
      report.steps.push(portraitStep);
      if (!portraitStep.ok) failures.push('portrait output did not work everywhere');
      select.value = 'landscape';
      select.dispatchEvent(new Event('change'));
      await wait(500);
    }

    report.ok = failures.length === 0;
    if (failures.length) report.error = failures.join('; ');
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
