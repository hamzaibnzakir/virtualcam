`face_landmarker.task` is Google's MediaPipe Face Landmarker model (float16, version 1), downloaded from
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
(sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff). Licensed under the Apache License 2.0.
It is used by the Lite engine to track 478 face points; it runs on the CPU (or GPU) with no graphics memory needed.

`selfie_multiclass_256x256.tflite` is Google's MediaPipe multiclass selfie segmentation model (float32), downloaded from
https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite
(16,371,837 bytes, sha256 c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0). Licensed under the Apache License 2.0.
It labels each pixel as background, hair, body skin, face skin, clothes or other (glasses, jewellery). The Lite engine
runs it once on the reference photo, so the photo's hair and glasses aren't painted onto your face.
