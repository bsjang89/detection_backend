from pathlib import Path
import cv2
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
import time

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Sources\Yolov8\runs\detect\train4\weights\best.pt" 
SOURCE_DIR = r"D:\Datasets\Otoki\All"

OUT_DIR = Path(r"C:\Sources\Yolov8\runs\infer\otoki_bb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 768
BATCH = 16
CONF_TH = 0.3
IOU_TH = 0.7
MAX_DET = 300
DEVICE = 0

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
}

# =========================
# DRAW + SAVE (병렬 대상)
# =========================
def draw_and_save(r):
    img = r.orig_img.copy()
    out_path = OUT_DIR / Path(r.path).name

    if r.boxes is None or len(r.boxes) == 0:
        #cv2.imwrite(str(out_path), img)
        cv2.imwrite(
        str(out_path.with_suffix(".jpg")),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90])
        return
    

    for box in r.boxes:
        conf = float(box.conf)
        cls_id = int(box.cls)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{r.names[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 6)
        cv2.putText(
            img,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA
        )

    #cv2.imwrite(str(out_path), img)
    cv2.imwrite(
    str(out_path.with_suffix(".jpg")),
    img,
    [cv2.IMWRITE_JPEG_QUALITY, 90]
    )


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    t0 = time.perf_counter()
    # 🔹 YOLO 추론 (GPU, batch)
    results = model.predict(
        source=SOURCE_DIR,
        imgsz=IMG_SIZE,
        batch=BATCH,
        conf=CONF_TH,
        iou=IOU_TH,
        max_det=MAX_DET,
        device=DEVICE,
        half=True,
        agnostic_nms=True,
        save=False,
        show=False,
        verbose=True,
    )

    t1 = time.perf_counter()
    print(f"Prediction: {(t1 - t0)*1000:.2f} ms")

    t0 = time.perf_counter()
    # 🔹 병렬 draw + save
    workers = max(1, cpu_count() // 4)
    with Pool(workers) as p:
        p.map(draw_and_save, results)

    t1 = time.perf_counter()
    print(f"Image Saved: {(t1 - t0)*1000:.2f} ms")

    print("Prediction done.")
