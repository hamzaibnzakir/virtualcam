// Scene data model, defaults and loading of saved state.

export const OUTPUT_WIDTH = 1280;
export const OUTPUT_HEIGHT = 720;
export const OUTPUT_FPS = 30;

export const SOURCE_TYPES = ['webcam', 'image', 'video', 'screen', 'card'];
export const AI_MODES = ['restyle', 'edit'];
export const AI_ENGINES = ['cloud', 'local', 'lite'];

// Used for the cloud AI when a reference photo is set and no look is typed.
export const CHARACTER_PROMPT =
  'Turn the person into the character from the reference image: same face, hair and outfit. Keep their movements and expressions.';

export const PRESETS = [
  { id: 'anime', label: 'Anime', prompt: 'Anime style, clean line art, vibrant cel shading' },
  { id: 'cyber', label: 'Cyberpunk', prompt: 'Cyberpunk style, neon lighting, futuristic city glow' },
  { id: 'oil', label: 'Oil painting', prompt: 'Classic oil painting, visible brush strokes, warm tones' },
  { id: 'studio', label: 'Studio lighting', prompt: 'Professional studio portrait lighting, soft key light, clean background' },
  { id: 'clay', label: 'Claymation', prompt: 'Claymation style, handmade clay figure, soft stop-motion look' },
];

let counter = 0;
export function newId() {
  counter += 1;
  return `s_${Date.now().toString(36)}_${counter.toString(36)}`;
}

export function makeScene(overrides = {}) {
  const base = {
    id: newId(),
    name: 'New scene',
    mirror: false,
    source: { type: 'webcam', deviceId: '', filePath: '', text: 'Be right back', color: '#1B1E27' },
    ai: { enabled: false, engine: 'cloud', mode: 'restyle', prompt: '', enhance: true, referencePath: '' },
    nameTag: { enabled: false, name: '', title: '' },
  };
  return normalizeScene({
    ...base,
    ...overrides,
    source: { ...base.source, ...(overrides.source || {}) },
    ai: { ...base.ai, ...(overrides.ai || {}) },
    nameTag: { ...base.nameTag, ...(overrides.nameTag || {}) },
  });
}

export function defaultState() {
  const scenes = [
    makeScene({ name: 'Real me' }),
    makeScene({
      name: 'Anime me',
      ai: { enabled: true, mode: 'restyle', prompt: PRESETS[0].prompt },
    }),
    makeScene({
      name: 'Clean office',
      ai: { enabled: true, mode: 'edit', prompt: 'Replace the background with a bright, modern office. Keep the person unchanged.' },
    }),
    makeScene({
      name: 'Become someone',
      ai: { enabled: true, engine: 'local', mode: 'edit', prompt: '' },
    }),
    makeScene({
      name: 'Be right back',
      source: { type: 'card', text: 'Be right back', color: '#1B1E27' },
    }),
  ];
  return { version: 1, scenes, activeId: scenes[0].id, settings: defaultSettings() };
}

function defaultSettings() {
  return { pricePerSecond: 0, faceBoost: false, cameraOutput: true };
}

function str(value, fallback = '', max = 2000) {
  return typeof value === 'string' ? value.slice(0, max) : fallback;
}

function isHexColor(value) {
  return typeof value === 'string' && /^#[0-9a-fA-F]{6}$/.test(value);
}

export function normalizeScene(raw) {
  const s = raw && typeof raw === 'object' ? raw : {};
  const src = s.source && typeof s.source === 'object' ? s.source : {};
  const ai = s.ai && typeof s.ai === 'object' ? s.ai : {};
  const tag = s.nameTag && typeof s.nameTag === 'object' ? s.nameTag : {};
  return {
    id: str(s.id) || newId(),
    name: str(s.name, 'Scene', 60) || 'Scene',
    mirror: Boolean(s.mirror),
    source: {
      type: SOURCE_TYPES.includes(src.type) ? src.type : 'webcam',
      deviceId: str(src.deviceId),
      filePath: str(src.filePath),
      text: str(src.text, 'Be right back', 120),
      color: isHexColor(src.color) ? src.color : '#1B1E27',
    },
    ai: {
      enabled: Boolean(ai.enabled),
      engine: AI_ENGINES.includes(ai.engine) ? ai.engine : 'cloud',
      mode: AI_MODES.includes(ai.mode) ? ai.mode : 'restyle',
      prompt: str(ai.prompt, '', 1000),
      enhance: ai.enhance !== false,
      referencePath: str(ai.referencePath),
    },
    nameTag: {
      enabled: Boolean(tag.enabled),
      name: str(tag.name, '', 60),
      title: str(tag.title, '', 80),
    },
  };
}

export function normalizeState(raw) {
  if (!raw || typeof raw !== 'object' || !Array.isArray(raw.scenes) || raw.scenes.length === 0) {
    return defaultState();
  }
  const seen = new Set();
  const scenes = raw.scenes.slice(0, 50).map(normalizeScene).filter((s) => {
    if (seen.has(s.id)) return false;
    seen.add(s.id);
    return true;
  });
  if (scenes.length === 0) return defaultState();
  const activeId = scenes.some((s) => s.id === raw.activeId) ? raw.activeId : scenes[0].id;
  const rawSettings = raw.settings && typeof raw.settings === 'object' ? raw.settings : {};
  const price = Number(rawSettings.pricePerSecond);
  return {
    version: 1,
    scenes,
    activeId,
    settings: {
      ...defaultSettings(),
      pricePerSecond: Number.isFinite(price) && price >= 0 ? price : 0,
      faceBoost: Boolean(rawSettings.faceBoost),
      cameraOutput: rawSettings.cameraOutput !== false,
    },
  };
}

export function sourceLabel(type) {
  return { webcam: 'Webcam', image: 'Image', video: 'Video loop', screen: 'Screen', card: 'Text card' }[type] || 'Webcam';
}

export function sceneKind(scene) {
  const base = sourceLabel(scene.source.type);
  if (!scene.ai.enabled) return base;
  if (scene.ai.engine === 'local') return `${base} + local face swap`;
  if (scene.ai.engine === 'lite') return `${base} + lite face warp`;
  if (scene.ai.referencePath) return `${base} + cloud character`;
  return `${base} + cloud ${scene.ai.mode === 'edit' ? 'edit' : 'restyle'}`;
}

// Phone cameras show up as ordinary webcams; their names give them away.
export function isPhoneCamera(label) {
  return /camo|iphone|ipad|droidcam|ivcam|epoccam|iriun|android|pixel|galaxy|phone|continuity/i.test(label || '');
}

export function fileName(filePath) {
  if (!filePath) return '';
  const parts = filePath.split(/[\\/]/);
  return parts[parts.length - 1] || filePath;
}
