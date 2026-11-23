# Task for Hermes: switch on automatic updates for Brainbox VirtualCam

From version 1.1.0, Brainbox VirtualCam updates itself. Every few hours it checks the **public**
GitHub repo `hamzaibnzakir/brainbox-virtualcam-releases`, downloads new versions in the background,
and installs them on restart. That repo holds installers only; the code stays in the private repo
`hamzaibnzakir/virtualcam`. The workflow `.github/workflows/release.yml` in the private repo builds
and uploads each version. These steps wire it up once, then publish 1.1.0.

## Steps
1. **Create the public releases repo** (logged-in browser or `gh`):
   ```bash
   gh repo create hamzaibnzakir/brainbox-virtualcam-releases --public \
     --description "Installers and automatic updates for Brainbox VirtualCam (no source code)" --add-readme
   ```
2. **Create a fine-grained token** at github.com → Settings → Developer settings → Fine-grained tokens:
   - Name `brainbox-releases-publisher`, expiry 1 year
   - Repository access: **only** `hamzaibnzakir/brainbox-virtualcam-releases`
   - Permissions: **Contents: Read and write** (nothing else)
3. **Store it as a secret** in the private repo. Don't paste it anywhere else, and don't commit it:
   ```bash
   gh secret set RELEASES_TOKEN --repo hamzaibnzakir/virtualcam   # paste the token when asked
   ```
4. **Publish version 1.1.0**:
   ```bash
   git clone https://github.com/hamzaibnzakir/virtualcam.git && cd virtualcam   # or git pull
   git tag v1.1.0 && git push origin v1.1.0
   ```
   Then watch the "Release" workflow: `gh run watch --repo hamzaibnzakir/virtualcam`.
5. **Check the result**: https://github.com/hamzaibnzakir/brainbox-virtualcam-releases/releases/tag/v1.1.0
   should have `Brainbox-VirtualCam-Setup-1.1.0.exe`, `Brainbox-VirtualCam-Setup-1.1.0.exe.blockmap`
   and `latest.yml`. Download `latest.yml` and check it names version 1.1.0.

## Later releases (no action from Hamza)
Bump `"version"` in `package.json` (e.g. 1.1.1), commit, then `git tag v1.1.1 && git push origin v1.1.1`.
Installed apps pick it up within a few hours, or right away from Settings → "Check for updates".

## Report back
```
Releases repo URL:
Token: created (name, expiry), scoped to the releases repo only, Contents read/write: yes/no
Secret RELEASES_TOKEN set on hamzaibnzakir/virtualcam: yes/no
Release workflow run URL and result:
Files on the v1.1.0 release (names + sizes):
latest.yml version line:
Problems and fixes:
```
