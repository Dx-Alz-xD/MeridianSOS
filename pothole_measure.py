"""
Pothole detection + size measurement.

Detector : YOLOv8 (trained on the MWPD pothole dataset, boxes only).
Size     : each pothole is modelled as an ellipse inscribed in its bounding box.
           * pixel mode  : radius / area in pixels (always available)
           * scale mode  : --cm-per-px  (top-down photo, or a known scale at the pothole)
           * camera mode : --cam-height-m + --pitch-deg + --hfov-deg
                           back-projects the ellipse onto the flat road plane, so the
                           perspective (far potholes look smaller) is handled properly.

"radius" reported = equivalent radius, i.e. radius of the circle with the same area.
Also reported: length / width (major / minor axis) of the ellipse.
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


# ------------------------------------------------------------------ geometry
def ellipse_points(x1, y1, x2, y2, n=90):
    """Points on the ellipse inscribed in the box (pixel coordinates)."""
    cx, cy, a, b = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([cx + a * np.cos(t), cy + b * np.sin(t)], axis=1)


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def pixel_to_ground(pts, img_w, img_h, cam_height, pitch_deg, hfov_deg):
    """
    Pinhole camera at height `cam_height` looking forward and tilted DOWN by `pitch_deg`
    (0 = looking at the horizon). Returns ground coordinates (X right, Y forward) in the
    same unit as cam_height. Points at/above the horizon return NaN.
    """
    f = (img_w / 2) / math.tan(math.radians(hfov_deg) / 2)
    th = math.radians(pitch_deg)
    dx = (pts[:, 0] - img_w / 2) / f
    dy = (pts[:, 1] - img_h / 2) / f
    denom = math.cos(th) * dy + math.sin(th)          # -d_Z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(denom > 1e-6, cam_height / denom, np.nan)
    gx = t * dx
    gy = t * (-math.sin(th) * dy + math.cos(th))
    return np.stack([gx, gy], axis=1)


def measure_box(box, img_w, img_h, cm_per_px=None, cam_height_m=None, pitch_deg=None, hfov_deg=70.0):
    """
    box = (x1, y1, x2, y2) in pixels.
    Returns dict with radius / area / length / width. Units: cm & cm2 if a scale is known,
    otherwise px & px2 (see 'unit').
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    if cam_height_m is not None and pitch_deg is not None:
        g = pixel_to_ground(ellipse_points(x1, y1, x2, y2), img_w, img_h, cam_height_m, pitch_deg, hfov_deg) * 100.0
        if np.isnan(g).any():
            return None                                   # box reaches the horizon, cannot be measured
        area = poly_area(g)
        d = np.sqrt(((g[:, None, :] - g[None, :, :]) ** 2).sum(-1))
        length = float(d.max())
        width = 2 * area / (math.pi * (length / 2)) if length > 0 else 0.0   # minor axis of equal-area ellipse
        unit, mode = "cm", "camera"
    else:
        s = cm_per_px if cm_per_px else 1.0
        length, width = max(x2 - x1, y2 - y1) * s, min(x2 - x1, y2 - y1) * s
        area = math.pi * (length / 2) * (width / 2)
        unit, mode = ("cm", "scale") if cm_per_px else ("px", "pixel")
    return {"radius": math.sqrt(area / math.pi), "area": area, "length": length, "width": width,
            "unit": unit, "mode": mode}


# ------------------------------------------------------------------ inference
def fmt(m):
    if m is None:
        return "n/a"
    u = m["unit"]
    return f"r={m['radius']:.1f}{u} A={m['area']:.0f}{u}2"


def run(weights, source, out_dir, conf=0.25, imgsz=416, **cam):
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    files = sorted(p for p in ([src] if src.is_file() else src.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rows = []
    for p in files:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        r = model.predict(img, conf=conf, imgsz=imgsz, device="cpu", verbose=False)[0]
        for i, (bx, c) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())):
            m = measure_box(bx, w, h, **cam)
            x1, y1, x2, y2 = map(int, bx)
            cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2), ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, (0, 200, 255), 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{c:.2f} {fmt(m)}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            rows.append([p.name, i, round(c, 3), x1, y1, x2, y2] +
                        ([round(m[k], 2) for k in ("radius", "area", "length", "width")] + [m["unit"], m["mode"]] if m else ["", "", "", "", "", "unmeasurable"]))
        cv2.imwrite(str(out_dir / p.name), img)
    with open(out_dir / "potholes.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "id", "confidence", "x1", "y1", "x2", "y2", "radius", "area", "length", "width", "unit", "mode"])
        wr.writerows(rows)
    print(f"{len(files)} images, {len(rows)} potholes -> {out_dir}/potholes.csv (+ annotated images)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Detect potholes and measure radius/area")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--source", required=True, help="image or folder of images")
    ap.add_argument("--out", default="results")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--cm-per-px", type=float, default=None, help="constant scale (top-down photos)")
    ap.add_argument("--cam-height-m", type=float, default=None, help="camera height above road (m)")
    ap.add_argument("--pitch-deg", type=float, default=None, help="camera tilt DOWN from horizontal (deg)")
    ap.add_argument("--hfov-deg", type=float, default=70.0, help="horizontal field of view (deg); phones ~65-75")
    a = ap.parse_args()
    run(a.weights, a.source, a.out, conf=a.conf, cm_per_px=a.cm_per_px,
        cam_height_m=a.cam_height_m, pitch_deg=a.pitch_deg, hfov_deg=a.hfov_deg)
