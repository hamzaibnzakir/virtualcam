// Studio window: scenes, live preview, AI controls and the output window.
import { DecartAI } from './engine/ai-decart.js';
import { LiteFaceAI } from './engine/ai-lite.js';
import { LocalFaceAI } from './engine/ai-local.js';
import { LookLayers } from './engine/look-layers.js';
import { VoiceStudio, VOICES } from './engine/voice.js';
import { voiceParams } from './engine/voice-dsp.js';
import { CallLink } from './engine/call-link.js';
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
// Hair and outfit from the photo, shared by the Lite and Local engines.
const look = new LookLayers(api);
local.look = look;
let lookWanted = { hair: false, outfit: false };
// Lite draws with WebGL; with hair or outfit on, its picture is copied here and they are added on top.
const liteView = document.createElement('canvas');
const liteViewCtx = liteView.getContext('2d', { alpha: false });

let liteUsesView = false;

function composeLite() {
  if (!liteUsesView) return;
  const w = lite.canvas.width;
  const h = lite.canvas.height;
  if (liteView.width !== w || liteView.height !== h) {
    liteView.width = w;
    liteView.height = h;
  }
  liteViewCtx.drawImage(lite.canvas, 0, 0);
  const t = performance.now();
  if (lookWanted.outfit) look.drawOutfit(liteViewCtx, lite.input, w, h, t, lite.points, { hideHair: lookWanted.hideHair });
  if (lookWanted.hideHair && lite.points) look.hideOwnHair(liteViewCtx, lite.input, w, h, t, lite.points, { segmented: lookWanted.outfit });
  if (lookWanted.hair && lite.points) look.drawHair(liteViewCtx, lite.points, lite.smoother.strength);
}

compositor.beforeRender = (drawSource) => {
  if (activeEngine === 'local') local.tick(drawSource);
  else if (activeEngine === 'lite') {
    lite.tick(drawSource);
    composeLite();
  }
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
  let engine = wantAi ? scene.ai.engine : 'none';
  // Local: the AI face swap (with Lite's tracking), or Lite's quick warp until the AI models are downloaded.
  if (engine === 'local' && !models.core) engine = 'lite';
  if (engine !== activeEngine) {
    activeEngine = engine;
    aiState = engine === 'cloud' ? cloudState : engine === 'local' ? local.state : engine === 'lite' ? lite.state : 'off';
  }
  compositor.aiFeed = engine === 'cloud';
  if (engine === 'cloud') compositor.setAiSource(aiVideo, () => cloudState === 'generating');
  else if (engine === 'local') compositor.setAiSource(local.canvas, () => local.live);
  else if (engine === 'lite') {
    liteUsesView = scene.ai.lookHair || scene.ai.lookOutfit;
    compositor.setAiSource(liteUsesView ? liteView : lite.canvas, () => lite.live);
  }
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

  // Hair and outfit from the photo (Lite and Local only; the cloud engine redraws everything itself).
  const wantLook = (engine === 'lite' || engine === 'local') && refBlob && (scene.ai.lookHair || scene.ai.lookOutfit);
  lookWanted = { hair: false, outfit: false };
  local.lookOpts = lookWanted;
  if (wantLook) {
    look
      .preparePhoto(refKey, refBlob, { outfit: scene.ai.lookOutfit || (scene.ai.lookHair && scene.ai.hideOwnHair) })
      .then((found) => {
        if (token !== applyToken) return;
        lookWanted = {
          hair: scene.ai.lookHair && look.hasHair,
          outfit: scene.ai.lookOutfit && look.hasOutfit,
          hideHair: scene.ai.lookHair && look.hasHair && scene.ai.hideOwnHair,
        };
        local.lookOpts = lookWanted;
        if (scene.ai.lookHair && found && !look.hasHair) showToast('No hair found in that photo to copy.');
        if (scene.ai.lookOutfit && found && !look.hasOutfit) showToast('No clothes found in that photo to copy. A photo showing the shoulders and chest works best.');
        renderAi();
      })
      .catch((err) => showToast(`Hair and outfit: ${(err && err.message) || err}`, true));
  }

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
  $('lite-group').hidden = !(scene.ai.engine === 'local' && !models.core);
  $('ai-mode').value = scene.ai.referencePath ? 'edit' : scene.ai.mode;
  $('ai-mode').disabled = Boolean(scene.ai.referencePath);
  setPressed($('toggle-enhance'), scene.ai.enhance);
  $('look-group').hidden = !(scene.ai.engine === 'lite' || scene.ai.engine === 'local');
  $('cloud-look-note').hidden = scene.ai.engine !== 'cloud';
  setPressed($('toggle-look-hair'), scene.ai.lookHair);
  setPressed($('toggle-look-outfit'), scene.ai.lookOutfit);
  setPressed($('toggle-hide-hair'), scene.ai.hideOwnHair);
  $('hide-hair-row').hidden = !scene.ai.lookHair;

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
    ? isLocal && !models.core
      ? 'Quick warp for now: you wear this face. Download the AI models for the full face swap.'
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
  const busyFast = download && download.pack === 'fast';
  $('local-model').value = state.settings.localModel;
  $('local-model').disabled = Boolean(busyFast);
  $('fast-line').hidden = !busyFast && !(state.settings.localModel === 'fast' && !models.fast);
  if (busyFast) {
    const pct = download.total ? Math.floor((download.done / download.total) * 100) : 0;
    $('fast-line').textContent = `Downloading the fast model… ${pct}%`;
  } else {
    $('fast-line').textContent = 'The fast model is not downloaded yet.';
  }
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
  local.swapFile = state && state.settings.localModel === 'fast' && models.fast ? models.fastFile : null;
  renderLocal();
  if (state) {
    applyAiVoices();
    renderVoice();
  }
}

async function startDownload(pack) {
  if (download) {
    api.modelsCancel();
    return;
  }
  download = { pack, done: 0, total: 1, file: 'models' };
  renderLocal();
  renderVoice();
  try {
    models = await api.modelsDownload(pack);
    showToast({ core: 'Local AI models downloaded', boost: 'Face detail boost downloaded', fast: 'Fast model downloaded', voices: 'AI voices downloaded' }[pack]);
  } catch (err) {
    showToast((err && err.message) || 'Download failed', true);
  } finally {
    download = null;
    await refreshModels();
    renderVoice();
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
  doc.title = 'Brainbox VirtualCam Output';
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
  if (!cameraState.installed) btn.textContent = cameraState.needsUpdate ? 'Update the camera' : 'Install the camera';
  else if (!state || !state.settings.cameraOutput) btn.textContent = 'Camera paused';
  else if (cameraState.inUse) btn.textContent = 'Camera: in use';
  else if (cameraState.running) btn.textContent = 'Camera: ready';
  else btn.textContent = 'Camera: starting…';

  const detail = $('camera-detail');
  if (!cameraState.supported) detail.textContent = 'The camera device is Windows only for now. Use the output window with OBS.';
  else if (cameraState.message) detail.textContent = cameraState.message;
  else if (cameraState.installed) detail.textContent = 'Installed. Pick "Brainbox VirtualCam" as the camera in Zoom, Meet, Teams, Discord or OBS.';
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
    showToast('Brainbox VirtualCam camera installed. Pick it in your video app.');
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
  renderPrefs();
  renderUpdates();
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

// ---------- always ready: tray, start with Windows, asking when a camera turns on ----------

function prefsOf(settings) {
  return {
    runInBackground: settings.runInBackground,
    startWithWindows: settings.startWithWindows,
    askOnCameraUse: settings.askOnCameraUse,
  };
}

function renderPrefs() {
  setPressed($('toggle-ask'), state.settings.askOnCameraUse);
  setPressed($('toggle-bg'), state.settings.runInBackground);
  setPressed($('toggle-login'), state.settings.startWithWindows);
}

function setPref(name, value) {
  state.settings[name] = value;
  renderPrefs();
  api.applyPrefs(prefsOf(state.settings));
  changed();
}

const lastAsked = new Map(); // app -> time, so the same app isn't asked about twice in a row

function askAboutCamera(event) {
  if (!state.settings.askOnCameraUse || !event) return;
  const who = event.kind === 'ours' ? 'ours' : String(event.app || 'An app');
  const now = Date.now();
  if (now - (lastAsked.get(who) || 0) < 10 * 60 * 1000) return;
  lastAsked.set(who, now);
  const scene = activeScene();
  const lookOn = scene.ai.enabled;
  const hasPhoto = Boolean(scene.ai.referencePath);
  if (event.kind === 'ours') {
    const actions = lookOn
      ? [
          { id: 'keep', label: 'Keep it on', primary: true },
          { id: 'look-off', label: 'Turn my look off' },
          { id: 'open', label: 'Open studio' },
        ]
      : hasPhoto || scene.ai.engine === 'cloud'
        ? [
            { id: 'look-on', label: 'Turn my look on', primary: true },
            { id: 'keep', label: 'Keep it plain' },
            { id: 'open', label: 'Open studio' },
          ]
        : [
            { id: 'open', label: 'Open studio', primary: true },
            { id: 'keep', label: 'Keep it plain' },
          ];
    api.showPrompt({
      title: 'A video app just turned on Brainbox VirtualCam',
      body: `Showing "${scene.name}". Your look is ${lookOn ? 'ON' : 'OFF'}. Do you want to use it now?`,
      actions,
      timeoutMs: 60000,
    });
    return;
  }
  const installed = cameraState.installed;
  api.showPrompt({
    title: `${who} turned on your real camera`,
    body: installed
      ? `Want Brainbox VirtualCam instead? In ${who}, pick "Brainbox VirtualCam" as the camera. Now showing "${scene.name}" (look ${lookOn ? 'on' : 'off'}).`
      : `Want Brainbox VirtualCam instead? Install its camera first (one click in the studio), then pick it in ${who}.`,
    actions: [
      { id: 'open', label: 'Open studio', primary: true },
      { id: 'dismiss', label: 'Not now' },
    ],
    timeoutMs: 45000,
  });
}

function handlePromptAction(id) {
  const scene = activeScene();
  if (id === 'look-on') {
    scene.ai.enabled = true;
    changed();
    showToast(`Look on for "${scene.name}"`);
  } else if (id === 'look-off') {
    scene.ai.enabled = false;
    changed();
    showToast(`Look off for "${scene.name}"`);
  }
}

// ---------- voice & calls ----------

const voice = new VoiceStudio({ api, onState: () => renderVoice() });
voice.testTone = Boolean(api.smoke);
let callVideoTrack = null;
const calls = new CallLink({
  api,
  // The same picture as the output window and the camera.
  // The Brainbox picture, or your real camera when "My face" is off.
  videoTrack: () => callVideo(),
  // A call that connects turns the microphone on (through the chosen voice).
  audioTrack: async () => {
    if (!voice.running) await voice.start();
    return callAudio();
  },
  onState: () => renderVoice(),
});
let callOpen = false;

function renderVoice() {
  if (!state) return;
  const grid = $('voice-presets');
  // The presets, then your own voices.
  const choices = [...Object.entries(VOICES).map(([id, v]) => [id, v.label]), ...Object.entries(state.settings.myVoices).map(([id, v]) => [`mine:${id}`, v.name])];
  const key = choices.map((c) => c.join('=')).join('|');
  if (grid.dataset.key !== key) {
    grid.dataset.key = key;
    grid.textContent = '';
    for (const [id, label] of choices) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.dataset.voice = id;
      btn.textContent = label;
      btn.addEventListener('click', () => chooseVoice(id));
      grid.append(btn);
    }
  }
  grid.querySelectorAll('button').forEach((b) => setPressed(b, b.dataset.voice === state.settings.voice));
  $('btn-voice-mic').textContent = voice.running ? 'Turn off microphone' : 'Turn on microphone';
  const pitch = voice.speakerF0 ? ` Your speaking pitch: ${Math.round(voice.speakerF0)} Hz.` : '';
  const mine = state.settings.voice.startsWith('mine:');
  const kind = state.settings.voice === 'own' ? '' : voice.aiLive ? ' (AI voice)' : mine ? ' (getting ready: your own voice for now)' : ' (voice effect)';
  $('voice-line').textContent = voice.error
    ? `Microphone problem: ${voice.error}`
    : voice.running
      ? `Live: ${voiceLabel(state.settings.voice)}${kind}.${pitch}${voice.speakerF0 ? '' : ' Say a sentence so the voice can tune itself to you.'}`
      : 'Pick a voice, then turn on the microphone to try it.';
  setPressed($('toggle-monitor'), voice.monitor);
  renderAiVoice();
  renderMyVoices();
  renderSend();
  const connected = calls.connected;
  $('call-status').textContent = connected
    ? `Call window is getting ${state.settings.sendFace ? 'your Brainbox picture' : 'your real camera'} and ${state.settings.sendVoice ? voiceLabel(state.settings.voice).toLowerCase() : 'your natural voice'}`
    : callOpen
      ? 'Call window open. Start or join a call there.'
      : 'No call window open';
  $('call-status').className = `ai-status${connected ? ' on' : ''}`;
}

// The AI voices block: download, on/off, and what the engine is doing.
function renderAiVoice() {
  const installed = Boolean(models.voices);
  const downloading = download && download.pack === 'voices';
  const btn = $('btn-voice-ai');
  const status = $('voice-ai-status');
  const line = $('voice-ai-line');
  setPressed($('toggle-ai-voices'), installed && state.settings.aiVoices);
  $('toggle-ai-voices').disabled = !installed;
  btn.hidden = false;
  if (downloading) {
    const pct = Math.round((download.done / Math.max(1, download.total)) * 100);
    status.textContent = `Downloading AI voices… ${pct}%`;
    status.className = 'ai-status';
    line.textContent = `${download.file} · ${(download.done / 1e6).toFixed(0)} of ${(download.total / 1e6).toFixed(0)} MB`;
    btn.textContent = 'Cancel';
    return;
  }
  if (!installed) {
    status.textContent = 'AI voices not downloaded';
    status.className = 'ai-status';
    line.textContent = 'Until then each voice is a voice effect: instant, but less natural.';
    btn.textContent = 'Download AI voices';
    return;
  }
  btn.hidden = voice.aiSummary() !== 'error';
  btn.textContent = 'Try the AI voice again';
  const name = voiceLabel(state.settings.voice).toLowerCase();
  const summary = voice.aiSummary();
  const where = voice.ai && voice.ai.loaded ? ` on ${voice.ai.loaded.device}` : '';
  const text = {
    disabled: ['AI voices off: using voice effects', ''],
    idle: ['AI voices ready', 'Pick a voice above to use it.'],
    missing: ['This voice has no AI version yet', 'It plays as a voice effect.'],
    'mine-missing': ['Voice models needed', 'Open "Add a voice" to download them (one time), then pick this voice again.'],
    off: ['AI voices ready', voice.running ? '' : 'Turn on the microphone to hear it.'],
    loading: [`Starting the AI ${name}…`, 'The voice effect plays until it is ready (a few seconds).'],
    on: [`AI ${name} live${where}`, 'About half a second behind you, like a phone call.'],
    error: ['The AI voice could not start', `${voice.aiError} The voice effect plays instead.`],
  }[summary] || ['', ''];
  status.textContent = text[0];
  status.className = `ai-status${summary === 'on' ? ' on' : summary === 'error' ? ' err' : summary === 'loading' ? ' warn' : ''}`;
  line.textContent = text[1];
}

// AI voices need a one-time agreement on how they may be used.
function askVoiceConsent() {
  if (state.settings.voiceConsent) return Promise.resolve(true);
  const dlg = $('voice-consent');
  return new Promise((resolve) => {
    dlg.addEventListener(
      'close',
      () => {
        const yes = dlg.returnValue === 'yes';
        if (yes) {
          state.settings.voiceConsent = true;
          changed();
        }
        resolve(yes);
      },
      { once: true },
    );
    dlg.returnValue = '';
    dlg.showModal();
  });
}

function applyAiVoices() {
  voice.gpuCount = local.gpuCount || 0;
  voice.setMyVoices(state.settings.myVoices, models.myVoice);
  voice.setAiVoices(models.voices, state.settings.aiVoices && state.settings.voiceConsent);
}

// ---------- what calls get: face and/or voice ----------

let realCamTrack = null;

// Your real camera (not Brainbox's own camera device), for calls with "My face" off.
async function realCamera() {
  if (realCamTrack && realCamTrack.readyState === 'live') return realCamTrack;
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  const cam = devices.find((d) => d.kind === 'videoinput' && !/brainbox|virtualcam/i.test(d.label));
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { deviceId: cam ? { exact: cam.deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  realCamTrack = stream.getVideoTracks()[0];
  return realCamTrack;
}

async function callVideo() {
  if (!state.settings.sendFace) {
    try {
      return await realCamera();
    } catch {
      // no camera: fall back to the Brainbox picture
    }
  }
  if (realCamTrack) {
    realCamTrack.stop();
    realCamTrack = null;
  }
  if (!callVideoTrack || callVideoTrack.readyState !== 'live') callVideoTrack = $('program').captureStream(OUTPUT_FPS).getVideoTracks()[0];
  return callVideoTrack;
}

function callAudio() {
  return state.settings.sendVoice ? voice.track : voice.rawTrack || voice.track;
}

function renderSend() {
  const { sendFace, sendVoice } = state.settings;
  setPressed($('toggle-send-face'), sendFace);
  setPressed($('toggle-send-voice'), sendVoice);
  $('send-face-note').textContent = sendFace ? '(Brainbox picture)' : '(your real camera)';
  $('send-voice-note').textContent = sendVoice ? `(${voiceLabel(state.settings.voice).toLowerCase()})` : '(your natural voice)';
  $('send-summary').textContent =
    sendFace && sendVoice ? 'Face and voice both changed' : sendFace ? 'Face only: your natural voice' : sendVoice ? 'Voice only: your real face' : 'Nothing changed: your real face and voice';
}

async function setSend(key, on) {
  state.settings[key] = on;
  changed();
  renderSend();
  if (!calls.peers.size) return; // nothing to switch until a call app asks
  if (key === 'sendFace') await calls.replaceVideo(await callVideo());
  else await calls.replaceAudio(callAudio());
}

function voiceLabel(id) {
  if (id && id.startsWith('mine:')) {
    const v = state.settings.myVoices[id.slice(5)];
    return v ? v.name : 'My voice';
  }
  return (VOICES[id] || VOICES.own).label;
}

// ---------- audio helpers ----------

// Any audio or video file -> 16 kHz mono samples (at most 10 minutes).
async function decodeTo16k(bytes) {
  const ctx = new AudioContext();
  let buf;
  try {
    buf = await ctx.decodeAudioData(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  } finally {
    ctx.close();
  }
  const seconds = Math.min(buf.duration, 600);
  const off = new OfflineAudioContext(1, Math.ceil(seconds * 16000), 16000);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}

function wavBytes(samples, rate) {
  const b = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, t) => [...t].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF');
  b.setUint32(4, 36 + samples.length * 2, true);
  str(8, 'WAVEfmt ');
  b.setUint32(16, 16, true);
  b.setUint16(20, 1, true);
  b.setUint16(22, 1, true);
  b.setUint32(24, rate, true);
  b.setUint32(28, rate * 2, true);
  b.setUint16(32, 2, true);
  b.setUint16(34, 16, true);
  str(36, 'data');
  b.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i += 1) b.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  return new Uint8Array(b.buffer);
}

// Records the microphone until stop() is called. Resolves the recording's bytes.
async function recordMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { deviceId: state.settings.micId ? { exact: state.settings.micId } : undefined, echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const rec = new MediaRecorder(stream);
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const done = new Promise((resolve) => {
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      resolve(new Uint8Array(await new Blob(chunks).arrayBuffer()));
    };
  });
  rec.start(500);
  const started = Date.now();
  return { stop: () => rec.state !== 'inactive' && rec.stop(), done, seconds: () => (Date.now() - started) / 1000 };
}

const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

// ---------- my voices ----------

const READ_TEXT =
  "Hi there, this is my voice. I'm recording this so my computer can learn how I sound. " +
  'The quick brown fox jumps over the lazy dog, and then runs all the way home before the rain. ' +
  "Yesterday I went to the shop to buy bread, milk, eggs, and a few oranges. It was busier than usual, so I waited a while. " +
  "Do you think it'll be sunny this weekend? I hope so, because we're planning a trip to the beach with some friends. " +
  'Numbers are useful too: one, two, three, four, five, six, seven, eight, nine, ten. ' +
  "When I'm happy I sometimes speak a little faster, and when I'm tired I slow right down. " +
  "Thanks for listening! If you can hear this clearly, the recording is working well. Let's keep going for a bit longer. " +
  'A good conversation needs questions, answers, laughs, and pauses. Really? Yes, really. Okay, great, that should be enough.';

const addVoice = { pcm: null, recorder: null, timer: null, busy: false, readyId: null };

function renderMyVoices() {
  const list = $('myvoice-list');
  list.textContent = '';
  const entries = Object.entries(state.settings.myVoices).sort((a, b) => b[1].created - a[1].created);
  if (!entries.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'No voices yet';
    list.append(li);
  }
  for (const [id, v] of entries) {
    const li = document.createElement('li');
    const info = document.createElement('div');
    info.className = 'grow';
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = v.name;
    const ready = document.createElement('div');
    ready.className = 'ready';
    ready.textContent = `Ready · ${Math.round(v.seconds / 6) / 10} min of speech · ${{ great: 'very close match', good: 'good match', basic: 'basic match: add more speech for better' }[v.quality]}`;
    info.append(name, ready);
    const use = document.createElement('button');
    use.type = 'button';
    use.className = 'btn btn-small';
    const on = state.settings.voice === `mine:${id}`;
    use.textContent = on ? 'In use' : 'Use';
    use.disabled = on;
    use.addEventListener('click', () => chooseVoice(`mine:${id}`));
    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'btn btn-small';
    del.textContent = 'Delete';
    del.setAttribute('aria-label', `Delete ${v.name}`);
    del.addEventListener('click', async () => {
      await api.deleteMyVoice(id).catch(() => {});
      delete state.settings.myVoices[id];
      if (state.settings.voice === `mine:${id}`) chooseVoice('own');
      changed();
      applyAiVoices();
      renderVoice();
    });
    li.append(info, use, del);
    list.append(li);
  }
}

function chooseVoice(id) {
  state.settings.voice = id;
  voice.setVoice(id);
  changed();
  renderVoice();
  calls.replaceAudio(callAudio());
}

function avStatus(text, kind = '') {
  const el = $('av-status');
  el.hidden = !text;
  el.textContent = text;
  el.className = `ai-status${kind ? ` ${kind}` : ''}`;
}

function avRender() {
  const ok = Boolean(addVoice.pcm) && $('av-consent').checked && !addVoice.busy && !addVoice.recorder;
  $('av-make').disabled = !ok && !addVoice.readyId;
  $('av-make').textContent = addVoice.readyId ? 'Use it now' : 'Make voice';
  $('av-upload').disabled = addVoice.busy || Boolean(addVoice.recorder);
  $('av-record').disabled = addVoice.busy || Boolean(addVoice.recorder);
}

function openAddVoice() {
  Object.assign(addVoice, { pcm: null, recorder: null, busy: false, readyId: null });
  $('av-name').value = '';
  $('av-consent').checked = false;
  $('av-source').textContent = '';
  $('av-reading').hidden = true;
  $('av-progress').hidden = true;
  avStatus('');
  if (!voice.ai) avStatus("Voices from uploads aren't available in this build.", 'err');
  avRender();
  $('add-voice').showModal();
}

async function useRecording(bytes, from) {
  try {
    const pcm = await decodeTo16k(bytes);
    const seconds = pcm.length / 16000;
    addVoice.pcm = pcm;
    $('av-source').textContent = `${from}: ${fmtTime(seconds)} of audio${seconds < 45 ? '. That is short: 1 to 5 minutes sounds much more like the person.' : '.'}`;
    avStatus('');
  } catch {
    addVoice.pcm = null;
    avStatus("That file couldn't be read as audio. Try a .wav, .mp3 or .m4a file.", 'err');
  }
  avRender();
}

async function makeVoice() {
  if (addVoice.readyId) {
    chooseVoice(`mine:${addVoice.readyId}`);
    $('add-voice').close();
    return;
  }
  if (!addVoice.pcm || !$('av-consent').checked) return;
  addVoice.busy = true;
  avRender();
  const bar = $('av-bar');
  $('av-progress').hidden = false;
  try {
    if (!models.myVoice) {
      avStatus('Downloading the voice models (one time, about 210 MB)…');
      if (download) throw new Error('Another download is running. Try again when it finishes.');
      await startDownload('myvoice');
      if (!models.myVoice) throw new Error("The voice models couldn't be downloaded.");
    }
    const id = `v${Date.now().toString(36)}`;
    const name = $('av-name').value.trim() || `Voice ${Object.keys(state.settings.myVoices).length + 1}`;
    avStatus(`Getting "${name}" ready… (this takes about a minute)`, 'warn');
    bar.style.width = '0%';
    const r = await voice.ai.prepare(id, addVoice.pcm, (p) => {
      bar.style.width = `${Math.round(p * 100)}%`;
      avStatus(`Getting "${name}" ready… ${Math.round(p * 100)}%`, 'warn');
    });
    state.settings.myVoices[id] = { name, seconds: r.seconds, quality: r.quality, created: Date.now() };
    changed();
    applyAiVoices();
    renderVoice();
    addVoice.readyId = id;
    bar.style.width = '100%';
    avStatus(`Ready: "${name}" (${fmtTime(r.seconds)} of speech used). You can use it in calls and voice messages now.`, 'on');
  } catch (err) {
    avStatus((err && err.message) || "The voice couldn't be made.", 'err');
  } finally {
    addVoice.busy = false;
    avRender();
  }
}

// ---------- voice messages ----------

const vm = { recorder: null, timer: null, result: null };

async function vmToggle() {
  if (vm.recorder) {
    vm.recorder.stop();
    return;
  }
  try {
    vm.recorder = await recordMic();
  } catch (err) {
    $('vm-line').textContent = `Microphone problem: ${(err && err.message) || err}`;
    return;
  }
  $('btn-vm-record').textContent = 'Stop';
  $('vm-actions').hidden = true;
  $('vm-audio').hidden = true;
  $('vm-line').textContent = `Recording in ${state.settings.sendVoice ? voiceLabel(state.settings.voice) : 'your natural voice'}…`;
  vm.timer = setInterval(() => {
    $('vm-time').textContent = fmtTime(vm.recorder ? vm.recorder.seconds() : 0);
  }, 250);
  const bytes = await vm.recorder.done;
  clearInterval(vm.timer);
  vm.recorder = null;
  $('btn-vm-record').textContent = 'Record';
  $('btn-vm-record').disabled = true;
  $('vm-line').textContent = 'Changing the voice…';
  try {
    const pcm16 = await decodeTo16k(bytes);
    const { samples, rate } = await convertMessage(pcm16);
    vm.result = wavBytes(samples, rate);
    const audio = $('vm-audio');
    if (audio.src) URL.revokeObjectURL(audio.src);
    audio.src = URL.createObjectURL(new Blob([vm.result], { type: 'audio/wav' }));
    audio.hidden = false;
    $('vm-actions').hidden = false;
    $('vm-line').textContent = `Ready: ${fmtTime(samples.length / rate)} in ${state.settings.sendVoice ? voiceLabel(state.settings.voice) : 'your natural voice'}. Play it, then save it.`;
  } catch (err) {
    $('vm-line').textContent = `Couldn't change the voice: ${(err && err.message) || err}`;
  } finally {
    $('btn-vm-record').disabled = false;
  }
}

// The whole message at once: an AI voice converts it in one go (best quality); a voice effect is
// applied offline; "My voice" off keeps it as recorded.
async function convertMessage(pcm16) {
  if (!state.settings.sendVoice || state.settings.voice === 'own') return { samples: pcm16, rate: 16000 };
  const ai = voice.aiVoice;
  if (ai) {
    const r = await voice.ai.convert(ai, pcm16);
    voice.apply(); // the live voice goes back to the chosen one
    return { samples: r.pcm, rate: r.rate };
  }
  const off = new OfflineAudioContext(1, Math.ceil((pcm16.length / 16000) * 48000), 48000);
  await off.audioWorklet.addModule('voice-worklet.js');
  const buf = off.createBuffer(1, pcm16.length, 16000);
  buf.copyToChannel(pcm16, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  const node = new AudioWorkletNode(off, 'brainbox-voice', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
  node.port.postMessage(voiceParams(state.settings.voice, voice.speakerF0));
  src.connect(node).connect(off.destination);
  src.start();
  const out = await off.startRendering();
  return { samples: out.getChannelData(0), rate: 48000 };
}

async function fillMics() {
  const select = $('mic-select');
  const devices = await navigator.mediaDevices.enumerateDevices().catch(() => []);
  const mics = devices.filter((d) => d.kind === 'audioinput' && d.deviceId !== 'default' && d.deviceId !== 'communications');
  select.textContent = '';
  const def = document.createElement('option');
  def.value = '';
  def.textContent = 'Default microphone';
  select.append(def);
  mics.forEach((m, i) => {
    const opt = document.createElement('option');
    opt.value = m.deviceId;
    opt.textContent = m.label || `Microphone ${i + 1}`;
    select.append(opt);
  });
  select.value = mics.some((m) => m.deviceId === state.settings.micId) ? state.settings.micId : '';
}

async function openCallSite(target) {
  const ok = await api.openCall(target);
  if (!ok) {
    showToast('That link must start with https://', true);
    return;
  }
  callOpen = true;
  renderVoice();
}

// ---------- updates ----------

let updateState = { status: 'idle', version: '' };

function renderUpdates() {
  const u = updateState;
  const text = {
    dev: `Version ${u.version} (test build: updates are off)`,
    idle: `Version ${u.version}`,
    checking: `Version ${u.version} · checking for updates…`,
    latest: `Version ${u.version} · you have the latest`,
    downloading: `Downloading version ${u.available}${u.progress ? ` (${u.progress}%)` : ''}…`,
    ready: `Version ${u.available} is ready. Restart to update.`,
    error: `Version ${u.version} · couldn't check for updates${u.error ? ` (${u.error})` : ''}`,
  }[u.status] || `Version ${u.version}`;
  $('update-detail').textContent = text;
  const ready = u.status === 'ready';
  $('btn-update-check').textContent = ready ? 'Restart to update' : 'Check for updates';
  $('btn-update-check').disabled = u.status === 'checking' || u.status === 'downloading' || u.status === 'dev';
  const banner = $('btn-update-ready');
  banner.hidden = !ready;
  if (ready) banner.textContent = `Update to ${u.available}`;
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
      showToast('Picture shape changed. In an open call, pick Brainbox VirtualCam again; in Chrome or Meet, reload the page.');
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
    activeScene().ai.engine = value === 'local' ? 'local' : 'cloud';
    changed();
  });
  $('btn-models').addEventListener('click', () => startDownload('core'));
  $('btn-engine-log').addEventListener('click', () => api.showEngineLog());
  // Everything needed to tune speed and quality on this PC, as text to paste into a chat.
  $('btn-speed-report').addEventListener('click', async () => {
    const info = await api.appInfo().catch(() => ({}));
    const scene = activeScene();
    const report = {
      app: info && info.version,
      hardware: hardware && { verdict: hardware.verdict, gpus: hardware.gpus, ramGB: hardware.ramGB },
      engine: activeEngine,
      look: { hair: lookWanted.hair, outfit: lookWanted.outfit, hideHair: Boolean(lookWanted.hideHair), segMs: look.segMs ? Math.round(look.segMs) : null },
      output: { fps: compositor.fps, size: `${$('program').width}x${$('program').height}` },
      lite: { fps: lite.stats.fps, ms: lite.stats.ms, tracking: lite.delegate },
      local: {
        model: state.settings.localModel,
        boost: state.settings.faceBoost,
        swapsPerSecond: local.stats.fps,
        ms: local.stats.ms,
        stages: local.stats.stages,
        device: local.deviceLabel || null,
        tried: local.tried,
        tracker: Boolean(local.tracker),
      },
      source: scene && scene.source && scene.source.type,
      voice: { running: voice.running, voice: state.settings.voice, ai: voice.aiSummary() },
    };
    const text = `Brainbox VirtualCam speed report\n${JSON.stringify(report, null, 2)}`;
    try {
      await navigator.clipboard.writeText(text);
      showToast('Speed report copied. Paste it into the chat.');
    } catch {
      showToast("Couldn't copy the report", true);
    }
  });
  // Inspector tabs: Look (AI) and Picture (source, name tag).
  const showTab = (name) => {
    document.querySelectorAll('.tabs [role="tab"]').forEach((t) => t.setAttribute('aria-selected', String(t.dataset.tab === name)));
    document.querySelectorAll('.inspector [data-tab]:not([role="tab"])').forEach((el) => {
      el.hidden = el.dataset.tab !== name;
    });
  };
  document.querySelectorAll('.tabs [role="tab"]').forEach((t) => t.addEventListener('click', () => showTab(t.dataset.tab)));
  showTab('look');
  $('local-model').addEventListener('change', async (e) => {
    const want = e.target.value === 'fast' ? 'fast' : 'best';
    if (want === 'fast' && !models.fast) {
      await startDownload('fast');
      if (!models.fast) {
        e.target.value = 'best';
        return;
      }
    }
    state.settings.localModel = want;
    local.swapFile = want === 'fast' ? models.fastFile : null;
    changed();
    local.reload();
    showToast(want === 'fast' ? 'Switched to the fast model' : 'Switched to the best-quality model');
  });
  // Voice & calls
  voice.setMyVoices(state.settings.myVoices, false);
  voice.setVoice(state.settings.voice);
  voice.micId = state.settings.micId;
  $('btn-voice-mic').addEventListener('click', async () => {
    if (voice.running) voice.stop();
    else await voice.start();
    await calls.replaceAudio(callAudio());
    fillMics();
    renderVoice();
  });
  $('mic-select').addEventListener('change', async (e) => {
    state.settings.micId = e.target.value;
    changed();
    await voice.setMic(e.target.value);
    await calls.replaceAudio(callAudio());
  });
  $('toggle-monitor').addEventListener('click', () => {
    voice.setMonitor(!voice.monitor);
    renderVoice();
  });
  $('btn-add-voice').addEventListener('click', openAddVoice);
  $('av-name').addEventListener('input', avRender);
  $('av-consent').addEventListener('change', avRender);
  $('av-upload').addEventListener('click', async () => {
    const file = await api.pickFile('audio');
    if (!file) return;
    await useRecording(await api.readFile(file), fileName(file));
  });
  $('av-record').addEventListener('click', async () => {
    try {
      addVoice.recorder = await recordMic();
    } catch (err) {
      avStatus(`Microphone problem: ${(err && err.message) || err}`, 'err');
      return;
    }
    $('av-text').textContent = READ_TEXT;
    $('av-reading').hidden = false;
    avRender();
    addVoice.timer = setInterval(() => {
      if (addVoice.recorder) $('av-timer').textContent = `${fmtTime(addVoice.recorder.seconds())} recorded (aim for 1-3 minutes; read it twice if you finish early)`;
    }, 250);
    const bytes = await addVoice.recorder.done;
    clearInterval(addVoice.timer);
    addVoice.recorder = null;
    $('av-reading').hidden = true;
    await useRecording(bytes, 'Your recording');
  });
  $('av-stop').addEventListener('click', () => addVoice.recorder && addVoice.recorder.stop());
  $('av-make').addEventListener('click', makeVoice);
  $('av-cancel').addEventListener('click', () => {
    if (addVoice.recorder) addVoice.recorder.stop();
    $('add-voice').close();
  });
  $('toggle-send-face').addEventListener('click', () => setSend('sendFace', !state.settings.sendFace));
  $('toggle-send-voice').addEventListener('click', () => setSend('sendVoice', !state.settings.sendVoice));
  $('btn-vm-record').addEventListener('click', vmToggle);
  $('btn-vm-save').addEventListener('click', async () => {
    if (!vm.result) return;
    const stamp = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, '');
    const saved = await api.saveVoiceMessage(vm.result, `voice-message-${stamp}.wav`).catch((err) => {
      showToast((err && err.message) || 'Could not save', true);
      return null;
    });
    if (saved) showToast(`Saved ${fileName(saved)}`);
  });
  $('btn-vm-discard').addEventListener('click', () => {
    vm.result = null;
    $('vm-audio').hidden = true;
    $('vm-actions').hidden = true;
    $('vm-line').textContent = '';
    $('vm-time').textContent = '';
  });
  $('btn-voice-ai').addEventListener('click', async () => {
    if (models.voices) {
      voice.retryAi();
      return;
    }
    if (download && download.pack !== 'voices') {
      showToast('Another download is running. Try again when it finishes.', true);
      return;
    }
    if (!download && !(await askVoiceConsent())) return;
    await startDownload('voices');
  });
  $('toggle-ai-voices').addEventListener('click', async () => {
    if (!models.voices) return;
    const on = !(state.settings.aiVoices && state.settings.voiceConsent);
    if (on && !(await askVoiceConsent())) return;
    state.settings.aiVoices = on;
    changed();
    applyAiVoices();
    renderVoice();
  });
  document.querySelectorAll('#call-sites [data-site]').forEach((b) => b.addEventListener('click', () => openCallSite(b.dataset.site)));
  $('btn-call-url').addEventListener('click', () => openCallSite($('call-url').value.trim()));
  api.onCallClosed(() => {
    callOpen = false;
    if (realCamTrack) {
      realCamTrack.stop();
      realCamTrack = null;
    }
    renderVoice();
  });
  const meterLoop = () => {
    $('voice-meter').style.width = `${Math.round(voice.meter() * 100)}%`;
    setTimeout(meterLoop, 80);
  };
  meterLoop();
  fillMics();
  renderVoice();
  $('toggle-ask').addEventListener('click', () => setPref('askOnCameraUse', !state.settings.askOnCameraUse));
  $('toggle-bg').addEventListener('click', () => setPref('runInBackground', !state.settings.runInBackground));
  $('toggle-login').addEventListener('click', () => setPref('startWithWindows', !state.settings.startWithWindows));
  api.onPrefsSet((next) => {
    for (const [k, v] of Object.entries(next || {})) if (k in prefsOf(state.settings)) state.settings[k] = Boolean(v);
    renderPrefs();
    api.applyPrefs(prefsOf(state.settings));
    changed();
  });
  api.onCameraUse(askAboutCamera);
  api.onPromptAction(handlePromptAction);
  api.onUpdates((next) => {
    updateState = { ...updateState, ...next };
    renderUpdates();
  });
  const runUpdateButton = () => (updateState.status === 'ready' ? api.installUpdate() : api.checkUpdates());
  $('btn-update-check').addEventListener('click', runUpdateButton);
  $('btn-update-ready').addEventListener('click', () => api.installUpdate());
  $('toggle-look-hair').addEventListener('click', () => {
    activeScene().ai.lookHair = !activeScene().ai.lookHair;
    changed();
  });
  $('toggle-hide-hair').addEventListener('click', () => {
    activeScene().ai.hideOwnHair = !activeScene().ai.hideOwnHair;
    changed();
  });
  $('toggle-look-outfit').addEventListener('click', () => {
    activeScene().ai.lookOutfit = !activeScene().ai.lookOutfit;
    changed();
  });
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
    if (!window.confirm('Remove the Brainbox VirtualCam camera from Windows? Video apps will no longer see it.')) return;
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
    .filter((cam) => !/virtualcam/i.test(cam.label))
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
  api.applyPrefs(prefsOf(state.settings));
  api.appInfo().then((info) => {
    updateState = { ...updateState, version: info.version };
    $('app-version').textContent = `v${info.version}`;
    renderUpdates();
  });

  setInterval(() => {
    if (activeEngine === 'cloud' && cloudState === 'generating' && !override) aiSeconds += 1;
    renderCost();
    if (activeEngine === 'local' || activeEngine === 'lite') renderLocal();
    $('foot-fps').textContent = `Output ${compositor.fps} fps`;
  }, 1000);

  api.onModelsProgress((p) => {
    if (download && p.pack === download.pack) {
      download = { ...download, ...p };
      if (p.pack === 'voices') renderVoice();
      else renderLocal();
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
  refreshModels().then(() => applyScene());
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
      if (!step.ok) failures.push('the Brainbox VirtualCam camera did not show the studio picture');
    }

    // Lite face warp on a still photo of a face.
    if (api.smokeFiles && api.smokeFiles.face && api.smokeFiles.ref) {
      const scene = makeScene({
        name: 'Smoke lite warp',
        source: { type: 'image', filePath: api.smokeFiles.face },
        ai: { enabled: true, engine: 'local', referencePath: api.smokeFiles.ref },
      });
      // Local without its AI models runs Lite's quick warp: test that path by hiding the models for a moment.
      const realModels = models;
      models = { ...models, core: false };
      state.scenes.push(scene);
      selectScene(scene.id);
      const until = Date.now() + 60000;
      while (Date.now() < until && !(lite.state === 'generating' && lite.faceFound && lite.frame > 10)) await wait(250);
      models = realModels;
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

      // Hair and outfit from the photo on top of the local swap.
      activeScene().ai.lookHair = true;
      activeScene().ai.lookOutfit = true;
      changed();
      const lookUntil = Date.now() + 60000;
      while (Date.now() < lookUntil && !(lookWanted.hair || lookWanted.outfit)) await wait(250);
      await wait(1500);
      const lookStep = { step: 'look', photoRead: Boolean(look.photo), hair: look.hasHair, outfit: look.hasOutfit, wanted: { ...lookWanted }, cameraClothes: Boolean(look.camBox), measured: look.measured, camRuns: look.camRuns, camError: look.camError, camFound: look.camFound, camOn: look.camOn, camSize: look.camSize, camBox: look.camBox, photoBox: look.photo && look.photo.clothesBox, videoSeg: Boolean(look.videoSeg), hideHair: Boolean(lookWanted.hideHair), hairFillOn: look.hairFillOn, hairCells: look.hairCells };
      if (api.smokeFiles.lookOut) {
        const png = await new Promise((r) => local.canvas.toBlob(r, 'image/png'));
        lookStep.saved = await api.smokeSave(api.smokeFiles.lookOut, new Uint8Array(await png.arrayBuffer()));
      }
      report.steps.push(lookStep);

      // AI voice: with the stand-in voice models installed, the microphone must go through the voice
      // engine process and come back (the stand-in voice hums at 200 Hz).
      if (api.smokeFiles.voices) {
        state.settings.voiceConsent = true;
        state.settings.aiVoices = true;
        state.settings.voice = 'young-woman';
        await refreshModels();
        voice.setVoice('young-woman');
        if (!voice.running) await voice.start();
        const aiUntil = Date.now() + 90000;
        while (Date.now() < aiUntil && !voice.aiLive && voice.aiSummary() !== 'error') await wait(250);
        await wait(3000);
        const aiStats = voice.ai ? await voice.ai.stats() : null;
        const aiStep = { step: 'ai-voice', installed: Boolean(models.voices), state: voice.aiSummary(), error: voice.aiError, live: voice.aiLive, loaded: voice.ai && voice.ai.loaded, stats: aiStats, playback: voice.aiBuffer || null };
        aiStep.ok = aiStep.live && Boolean(aiStats && aiStats.steps > 0);
        report.steps.push(aiStep);
        if (!aiStep.ok) failures.push('the AI voice engine did not convert the microphone');

        // My voices: prepare a voice from a 40 s "upload", use it live, and convert a voice message.
        const mineStep = { step: 'my-voice' };
        try {
          const upload = new Float32Array(16000 * 40).map((_, i) => (0.2 + 0.15 * Math.sin(i / 3000)) * Math.sin(i * 0.11) * (Math.random() * 0.5 + 0.5));
          const prepared = await voice.ai.prepare('smoketest', upload, () => {});
          state.settings.myVoices.smoketest = { name: 'Smoke voice', seconds: prepared.seconds, quality: prepared.quality, created: Date.now() };
          applyAiVoices();
          chooseVoice('mine:smoketest');
          const mineUntil = Date.now() + 60000;
          while (Date.now() < mineUntil && !(voice.aiLive && voice.ai.voiceFile === 'mine:smoketest') && voice.aiSummary() !== 'error') await wait(250);
          let stats = null;
          const stepsUntil = Date.now() + 15000;
          while (Date.now() < stepsUntil) {
            await wait(1000);
            stats = await voice.ai.stats();
            if (stats && stats.engine === 'my-voice' && stats.steps > 0) break;
          }
          const message = await convertMessage(upload.subarray(0, 16000 * 3));
          Object.assign(mineStep, { prepared, live: voice.aiLive, state: voice.aiSummary(), error: voice.aiError, stats, messageSeconds: message.samples.length / message.rate });
          mineStep.ok = voice.aiLive && stats && stats.engine === 'my-voice' && stats.steps > 0 && mineStep.messageSeconds > 2.5;
        } catch (err) {
          mineStep.error = (err && err.message) || String(err);
          mineStep.ok = false;
        }
        report.steps.push(mineStep);
        if (!mineStep.ok) failures.push('My voices: preparing or using an uploaded voice failed');
      }

      // Calls window: a test page asks for a camera and microphone like a call app; it must get the
      // studio's picture and the (changed) voice.
      state.settings.voice = 'young-woman';
      voice.setVoice('young-woman');
      await api.openCall('smoke-test');
      const callUntil = Date.now() + 120000;
      let callResult = null;
      while (Date.now() < callUntil) {
        callResult = await api.smokeCallResult();
        if (callResult && callResult.stage !== 'starting') break;
        await wait(500);
      }
      const callStep = { step: 'calls', result: callResult, connected: calls.connected, voiceRunning: voice.running, voiceError: voice.error };
      callStep.ok = Boolean(callResult && callResult.stage === 'done' && callResult.brightness > 5 && callResult.audioLive && callResult.audioRms > 0.001 && callResult.video !== '0x0');
      report.steps.push(callStep);
      if (!callStep.ok) failures.push('the Calls window did not get the picture and voice');
      if (!look.photo) failures.push('hair/outfit: the photo could not be read');
    }


    // Portrait output: every engine and the camera follow the new shape; 'fit' shows the whole picture.
    if (api.smokeFiles && api.smokeFiles.face && api.smokeFiles.ref) {
      const select = $('output-shape');
      select.value = 'portrait';
      select.dispatchEvent(new Event('change'));
      const portraitStep = { step: 'portrait' };
      const liteScene = state.scenes.find((s) => s.name === 'Smoke lite warp');
      liteScene.fit = 'fit';
      const savedModels = models;
      models = { ...models, core: false }; // this scene runs as the quick warp (see the lite step)
      selectScene(liteScene.id);
      const startFrame = lite.frame;
      const until = Date.now() + 60000;
      while (Date.now() < until && lite.frame < startFrame + 5) await wait(250);
      models = savedModels;
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
  if (api.smokeFiles && api.smokeFiles.tab) {
    const [name, where] = api.smokeFiles.tab.split(':');
    const tab = document.querySelector(`[role="tab"][data-tab="${name}"]`);
    if (tab) tab.click();
    await wait(300);
    if (where === 'bottom') {
      const panels = document.querySelectorAll(`section[data-tab="${name}"]`);
      if (panels.length) panels[panels.length - 1].scrollIntoView({ block: 'end' });
    }
    await wait(500);
  }
  api.smokeReport(report);
}

window.addEventListener('error', (e) => showToast(`Something went wrong: ${e.message}`, true));
window.addEventListener('unhandledrejection', (e) => {
  const reason = e.reason;
  showToast(`Something went wrong: ${(reason && reason.message) || reason}`, true);
});

start().catch((err) => showToast(`Brainbox VirtualCam could not start: ${err.message}`, true));
