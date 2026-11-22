# VirtualCam

A desktop studio that decides what your camera shows. Drop in a photo and you become that person, following your real moves. Or show a looping video, a picture, your screen or a text card. Switch between "scenes" instantly with hotkeys from any app.

Zoom, Google Meet (in Chrome or Edge), Teams, Discord and OBS see it as a normal camera called **VirtualCam**, on Windows 10 and Windows 11.

## Install (Windows)

**Option A: installer.** Open this repo's **Actions** tab, then **App on Windows**, then the latest green run. Under **Artifacts**, download **VirtualCam-Windows-installer** and run it.

**Option B: from source.** Install Node.js LTS, then in this folder run:

```
npm install
npm start
```

Then, in the app:

1. Click **Install VirtualCam camera** (top bar). Windows asks for admin approval once.
2. In your video app, pick **VirtualCam** as the camera.

The app only spends effort on the camera while some app is actually showing it.

## Cameras: laptop or phone

Pick any camera in **Source → Camera**. Phone cameras are usually sharper, which also helps the AI. In the app, see **Use your phone as the camera**:

- **iPhone:** Camo (free, 720p) over the cable. Needs iTunes on the PC.
- **Android 14+:** plug in, then choose **USB → Webcam** on the phone. On phones without that option, use DroidCam.

## Becoming someone: the two AI engines

Drop a photo anywhere on the window (or click the photo box) and turn on **AI transform**.

| | **Local · face swap** | **Cloud · face and clothes** |
|---|---|---|
| What changes | Your face becomes their face. Hair, clothes, body and moves stay yours. | Face, hair and outfit become theirs. Your moves and expressions drive it. |
| Cost | Free, runs on your computer | Decart, $0.02 per second (about $1.20 per minute) |
| Needs | A one-time 470 MB model download. A graphics card with 2 GB+ is recommended (any DirectX 12 GPU; your Dell's Quadro T2000 qualifies). | A Decart API key (Settings) and a good internet connection |
| Extras | **Face detail boost**: sharper face, slower, 340 MB extra download | Type a look or pick a quick style instead of a photo |

The app checks your graphics card and says whether local AI will run well. If AI stops or drops, the real picture shows instead. It never goes black. Your **voice is never touched**: VirtualCam is video only, so your microphone goes straight to the call.

## Everyday use

- **Scenes:** click one, or press **Ctrl + Alt + 1–9** from any app.
- **Panic:** **Ctrl + Alt + 0**, or the red button, goes straight back to your plain camera.
- **Quick switcher:** **Ctrl + Alt + Space**.
- **Video loop / image / screen / text card:** choose them under **Source**.
- **Name tag:** an on-screen name and title.
- **Output window:** a clean 1280×720 window for OBS, or for Mac and Linux where the camera device isn't available yet.

## Using it responsibly

Use your own face, characters, or people who have agreed to it. Don't use it to pass as someone else, or to fool identity or selfie checks. VirtualCam has no features for that, and none will be added.

## How it's tested

- `npm run smoke` launches the app hidden with a fake camera and exercises every scene type, panic mode, the output window, the quick switcher, a cloud connection with a bad key, and (with the test models) a real local face swap. It fails on any console error.
- `node scripts/test-faceswap.cjs` checks swap quality with a face-recognition model: the swapped face must match the photo's identity.
- **GitHub Actions → Windows camera** builds the camera device (64-bit and 32-bit) and proves it end to end on Windows. It registers the camera, sends known colours, captures them back through DirectShow like a video app, and checks they arrive exactly. Log: `ci-logs/windows-camera.txt`.
- **GitHub Actions → App on Windows** runs the real app on Windows. It installs the camera through the app, reads the app's picture back from the VirtualCam camera, downloads the models with the app's downloader, runs a local face swap, and builds the installer. Log: `ci-logs/app-windows.txt`.

## Project layout

```
src/main/main.js              windows, hotkeys, files, encrypted key storage, IPC
src/main/camera-driver.js     installs the VirtualCam camera and streams frames into it
src/main/models.js            model downloads with checksums
src/main/hardware.js          graphics card / VRAM check
src/main/faceswap/            local face swap engine (runs in its own process)
  worker.js                   engine process
  swapper.js                  detect -> identity -> swap -> optional detail boost
  detector.js                 SCRFD face finder
  face-math.js, paste.js      alignment and blending
  onnx-initializer.js         reads the swap model's identity matrix
src/renderer/                 the studio window
  engine/ai-local.js          talks to the local engine, composites swapped faces
  engine/ai-decart.js         Decart realtime (cloud) adapter
  engine/compositor.js        draws the picture at 30 fps
native/softcam/               the camera device (Softcam, MIT, renamed to VirtualCam)
native/vcam-bridge/           tiny program that feeds frames into the camera
vendor/win/                   tested camera binaries, committed by CI
```

## Credits and licenses

- Camera device: [Softcam](https://github.com/tshino/softcam) by tshino, MIT.
- Face models are downloaded on first use. The InsightFace models (face finder, identity, inswapper) are licensed for **non-commercial use only**. GFPGAN is Apache-2.0. Selling VirtualCam with the local engine would need commercially licensed models.
- Cloud AI: [Decart](https://platform.decart.ai) realtime API.
