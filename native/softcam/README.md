# VirtualCam camera driver (based on Softcam)

This folder is [Softcam](https://github.com/tshino/softcam) by tshino (MIT license, see `LICENSE`),
vendored from commit d1acde4 with these changes only:

- The camera is named **VirtualCam** (was "DirectShow Softcam").
- It has its own class ID and shared-memory names, so it never clashes with other Softcam-based cameras.
- Visual Studio 2019 project files and the upstream tests/examples were left out.

It registers a DirectShow video capture device, which Windows 10 and Windows 11 apps that use
DirectShow (Zoom, Teams, Discord, OBS, Chrome/Edge-based apps) list as a camera.
It is built and tested by `.github/workflows/windows-camera.yml`.
