// The corner question card ("An app started using the camera...").
const api = window.vcam;

function render(data) {
  if (!data) return;
  document.getElementById('title').textContent = data.title;
  document.getElementById('body').textContent = data.body;
  const actions = document.getElementById('actions');
  actions.textContent = '';
  for (const a of data.actions || []) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `act${a.primary ? ' primary' : ''}`;
    btn.textContent = a.label;
    btn.addEventListener('click', () => api.promptAction(a.id));
    actions.append(btn);
  }
}

api.onPromptData(render);
document.getElementById('close').addEventListener('click', () => api.promptAction('dismiss'));
