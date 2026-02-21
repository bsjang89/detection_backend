from pathlib import Path
import cv2
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
import time
import numpy as np
from shapely.geometry import Polygon

# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:\Sources\Yolov8\runs\obb\train5\weights\best.pt"
SOURCE_DIR = r"D:\Datasets\Otoki\All"

OUT_DIR = Path(r"C:\Sources\Yolov8\runs\infer\otoki")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 640
BATCH = 16
##CONF_TH = 0.1
#IOU_TH = 0.7          # YOLO 내부 NMS
#OVERLAP_IOU = 0.1     # 커스텀 OBB 중복 제거 기준
#MAX_DET = 300
DEVICE = 0

CONF_TH = 0.01          # ⭐ 0.1 → 0.25
IOU_TH = 1          # 내부 정리용
OVERLAP_IOU = 0.1       # ⭐ 0.1 → 0.4
MAX_DET = 1000

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
}

# =========================
# OBB IoU
# =========================
def polygon_iou(p1, p2):
    poly1 = Polygon(p1)
    poly2 = Polygon(p2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return inter / union if union > 0 else 0.0


def suppress_obb(polys, confs, iou_th=0.5):
    idxs = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        i = idxs.pop(0)
        keep.append(i)

        idxs = [
            j for j in idxs
            if polygon_iou(polys[i], polys[j]) < iou_th
        ]

    return keep


# =========================
# DRAW + SAVE
# =========================
def draw_and_save_obb(r):
    img = r.orig_img.copy()
    out_path = OUT_DIR / Path(r.path).name

    if r.obb is None or len(r.obb) == 0:
        #cv2.imwrite(str(out_path), img)
        cv2.imwrite(
        str(out_path.with_suffix(".jpg")),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90])
        return

    polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    confs = r.obb.conf.cpu().numpy()
    clss  = r.obb.cls.cpu().numpy().astype(int)

    # 🔹 confidence 1차 필터
    valid = confs >= 0.1
    polys = polys[valid]
    confs = confs[valid]
    clss  = clss[valid]

    if len(polys) == 0:
        #cv2.imwrite(str(out_path), img)
        cv2.imwrite(
        str(out_path.with_suffix(".jpg")),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90])
        return

    # =========================
    # 🔹 IoU 디버그 putText
    # =========================
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            iou = polygon_iou(polys[i], polys[j])

            if iou >= 0.2:  # 👈 디버그 표시 기준 (조절 가능)
                cx = int((polys[i][:, 0].mean() + polys[j][:, 0].mean()) / 2)
                cy = int((polys[i][:, 1].mean() + polys[j][:, 1].mean()) / 2)

                cv2.putText(
                    img,
                    f"IoU {iou:.2f}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),  # 노랑
                    2,
                    cv2.LINE_AA,
                )

    # =========================
    # 🔹 중복 제거 (conf 높은 것 유지)
    # =========================
    keep_idxs = suppress_obb(polys, confs, iou_th=OVERLAP_IOU)

    for i in keep_idxs:
        pts = polys[i].astype(int)
        cls_id = clss[i]
        conf = confs[i]

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{r.names[cls_id]} {conf:.2f}"

        cv2.polylines(img, [pts], True, color, 6)

        # OBB 중심 기준 label
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        cv2.putText(
            img,
            label,
            (cx - 40, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

    #cv2.imwrite(str(out_path), img)
    cv2.imwrite(
        str(out_path.with_suffix(".jpg")),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, 90])


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    t0 = time.perf_counter()
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
    workers = max(1, cpu_count() // 4)
    with Pool(workers) as p:
        p.map(draw_and_save_obb, results)
    t1 = time.perf_counter()
    print(f"Image Saved: {(t1 - t0)*1000:.2f} ms")

    print("Prediction done.")
