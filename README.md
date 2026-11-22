# VirtualCam

A desktop studio that decides what your camera shows: your real webcam, an image, a looping video, your screen, a text card, or a live AI-restyled version of any of them. You switch between these "scenes" instantly with hotkeys, from any app.

This is **phase 1**: the studio, scene engine, AI and hotkeys. Phase 2 adds VirtualCam as a real camera device in Windows. Until then, OBS carries the picture into Zoom, Meet and similar apps (see "Use it in calls today").

## Run it (Windows or Mac)

1. Install **Node.js LTS** from nodejs.org.
2. Unzip this folder, open a terminal in it, and run:

   ```
   npm install
   npm start
   ```

3. Open **Settings** (the gear, top right) and paste your **Decart API key** (from platform.decart.ai). It is stored encrypted on your computer.

## Using it

- **Scenes** (left): click one, or press **Ctrl + Alt + 1–9** in any app (Cmd + Option on Mac). Number keys 1–9 also work while the studio is focused.
- **Panic**: **Ctrl + Alt + 0**, or the red button, goes straight back to your plain webcam with no AI and no overlays.
- **Quick switcher**: **Ctrl + Alt + Space** pops up a scene list over whatever you're doing.
- **Source** (right): Webcam, Image, Video (loops), Screen (the screen your mouse is on) or Text card.
- **AI transform**: turn it on, then type a look and press **Apply live**, or tap a quick style. The prompt changes live without reconnecting.
  - *Restyle me* restyles the whole picture.
  - *Edit scene or become a character* edits parts of the picture (background swaps, objects). It also accepts a **reference image**, so you can become the person or character in a photo.
- **Name tag**: an on-screen name and title.
- **Cost meter**: counts the seconds AI is actually generating. Add your provider's per-second price in Settings to see dollars.

If the AI connection drops, or the key is missing or wrong, the output falls back to the real picture. It never goes black.

## Use it in calls today (with OBS)

1. Click **Open output window** in VirtualCam. It shows a clean 1280×720 picture.
2. In OBS: add a **Window Capture** source and pick "VirtualCam Output".
3. Click **Start Virtual Camera** in OBS.
4. In Zoom, Meet, Discord etc., choose **OBS Virtual Camera** as your camera.

Keep the output window open (it can sit behind other windows). Minimizing the studio is fine, because frames keep rendering.

## Build an installer

```
npm run dist        # Windows .exe installer (run this on Windows)
npm run dist:mac    # Mac .dmg (run this on a Mac)
```

The installer lands in `release/`.

## Self-test

```
npm run smoke
```

This launches the app hidden with a fake camera and runs every scene type, panic mode, a new scene with a name tag, the output window, the quick switcher, and an AI connection with a bad key. It fails on any console error.

## Project layout

```
src/main/main.js            app windows, hotkeys, files, encrypted key storage
src/main/preload.js         the safe bridge between windows and the app
src/renderer/studio.*       the studio window
src/renderer/switcher.*     the quick switcher overlay
src/renderer/engine/
  state.js                  scene model, defaults, saved-state loading
  sources.js                webcam / image / video / screen / text card
  compositor.js             draws the program picture at 30 fps + name tag
  ai-decart.js              Decart realtime adapter (WebRTC in, video out)
```

AI providers share one small shape (`setTarget`, `stop`, state and error callbacks), so fal.ai or others can be added as another adapter next to `ai-decart.js`.

## What's next (phase 2)

- Register VirtualCam as a real Windows camera (Media Foundation virtual camera on Windows 11, DirectShow fallback for older apps), so OBS is no longer needed.
- Mac camera extension.
- A fal.ai adapter as a second AI provider.
