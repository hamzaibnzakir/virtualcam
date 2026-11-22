#!/usr/bin/env bash
# One-time setup on the rented GPU machine. Run from scripts/distill/.
#   bash setup_vast.sh            (models to /workspace/models, data to /workspace/data)
set -euo pipefail
W=${WORKSPACE:-/workspace}
mkdir -p "$W/models" "$W/data" "$W/runs"

pip install -q -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU visible'; print('GPU:', torch.cuda.get_device_name(0))"
python -c "import onnxruntime as o; p=o.get_available_providers(); print('onnxruntime:', p); assert 'CUDAExecutionProvider' in p"

fetch() { # url file sha256
  if [ ! -f "$W/models/$2" ]; then curl -fL --retry 3 -o "$W/models/$2" "$1"; fi
  echo "$3  $W/models/$2" | sha256sum -c -
}
fetch https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx inswapper_128.onnx e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af
fetch https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/det_10g.onnx det_10g.onnx 5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91
fetch https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx w600k_r50.onnx 4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43
fetch https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/genderage.onnx genderage.onnx 4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb

# Training faces: SFHQ (AI-generated faces, no real people). Parts 1 and 2 for training, part 3 for the
# final check (the app's own benchmark also uses part 3, so it never trains on it). Needs ~/.kaggle/kaggle.json.
cd "$W/data"
for part in 1 2 3; do
  if [ ! -d "sfhq$part" ]; then
    kaggle datasets download -d "selfishgene/synthetic-faces-high-quality-sfhq-part-$part" -p "sfhq$part" --unzip
  fi
done
echo "setup done"; df -h "$W" | tail -1
