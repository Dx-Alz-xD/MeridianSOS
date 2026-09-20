from ultralytics import YOLO
m = YOLO("yolov8n.pt")
m.train(data="/home/claude/pothole_ds/data.yaml", epochs=15, imgsz=416, batch=16, workers=1,
        device="cpu", project="/home/claude/runs", name="pothole", exist_ok=True,
        patience=8, close_mosaic=3, plots=False, seed=0, verbose=True)
