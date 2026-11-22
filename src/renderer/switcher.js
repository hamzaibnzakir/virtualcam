// Quick switcher overlay: number keys pick a scene, 0 goes back to the real camera.
const api = window.vcam;
const list = document.getElementById('list');
let snapshot = { scenes: [], activeId: null };

function render() {
  list.textContent = '';
  snapshot.scenes.slice(0, 9).forEach((scene, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    if (scene.id === snapshot.activeId) btn.setAttribute('aria-current', 'true');
    const kbd = document.createElement('kbd');
    kbd.textContent = String(i + 1);
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = scene.name;
    const kind = document.createElement('span');
    kind.className = 'kind';
    kind.textContent = scene.kind;
    btn.append(kbd, name, kind);
    btn.addEventListener('click', () => api.chooseScene(i));
    list.append(btn);
  });
  const current = list.querySelector('[aria-current="true"]') || list.querySelector('button');
  if (current) current.focus();
}

api.onSwitcherScenes((next) => {
  if (next && Array.isArray(next.scenes)) snapshot = next;
  render();
});

document.getElementById('panic').addEventListener('click', () => api.panic());

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    api.closeSwitcher();
  } else if (e.key === '0') {
    api.panic();
  } else if (/^[1-9]$/.test(e.key)) {
    const index = Number(e.key) - 1;
    if (index < snapshot.scenes.length) api.chooseScene(index);
  }
});
