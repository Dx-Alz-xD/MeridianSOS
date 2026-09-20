# Pothole detection + radius / area

Files
- best.pt              trained YOLOv8n pothole detector
- pothole_measure.py   runs the detector and measures radius / area / length / width
- make_split.py        leak-free dataset split (grouped by source photo)
- train.py             training script used (CPU, 416 px, 15 epochs)
- demo_output/         annotated example images + potholes.csv (pixel units)

Install:  pip install ultralytics opencv-python

Run (pixel units, no setup):
  python pothole_measure.py --weights best.pt --source photo.jpg
Run (real units, camera known):
  python pothole_measure.py --weights best.pt --source photos/ --cam-height-m 1.2 --pitch-deg 25 --hfov-deg 70
Run (top-down photo, known scale):
  python pothole_measure.py --weights best.pt --source photo.jpg --cm-per-px 0.4

Held-out test set (102 images, no overlap with training photos)
  precision 0.66, recall 0.60, mAP50 0.66, mAP50-95 0.31
Size accuracy vs labelled boxes (136 matched detections)
  median radius error 8%, median area error 16%, 69% of areas within +-25%

Limits
- The dataset has boxes only, so a pothole is modelled as the ellipse inscribed in its box.
  Radius = radius of the circle with the same area. It does not follow the true outline.
- cm values are only as good as the camera height / tilt / field of view (or cm-per-px) you give.
  The dataset mixes dashcam, roadside and top-down photos, so one setting does not fit all;
  the demo is therefore in pixels. Measure with your own camera setup for real units.
- Depth is not measured, only surface size. Trained on a small CPU run; a GPU run with a
  larger model (yolov8s, 640 px, 50+ epochs) should improve recall.
