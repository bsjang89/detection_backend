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
MODEL_PATH = r"C:\Sources\Yolov8\runs\obb\train7\weights\best.pt"
SOURCE_DIR = r"D:\Datasets\Otoki\All"

OUT_DIR = Path(r"C:\Sources\Yolov8\runs\infer\otoki_3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 768
BATCH = 16
DEVICE = 0

# ---- Prediction thresholds (YOLO internal)
CONF_TH_PRED = 0.3      # 디버그용: 너무 낮게(0.001) 두면 FP 폭주가 정상입니다.
IOU_TH_PRED  = 0.5      # 내부 NMS 활성화(정상 범위)
MAX_DET      = 300

# ---- Drawing thresholds (must be consistent)
CONF_TH_DRAW = CONF_TH_PRED

# ---- Custom OBB overlap suppression (post-filter)
OVERLAP_IOU = 0.50       # 0.1은 너무 빡세서 이상한 FP만 남는 경우가 많음

# ---- Debug options
SAVE_ULTRA_PLOT = True   # Ultralytics 기본 plot도 저장(좌표 문제 vs FP 구분용)
ULTRA_PLOT_DIR = OUT_DIR / "_ultra_plot"
ULTRA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 0),
}

def obb_area(pts):
    # pts: (4,2)
    return Polygon(pts).area


# =========================
# OBB IoU
# =========================
def polygon_iou(p1, p2):
    """
    p1, p2: (4,2) numpy arrays in pixel coords
    """
    poly1 = Polygon(p1)
    poly2 = Polygon(p2)

    if (not poly1.is_valid) or (not poly2.is_valid):
        return 0.0

    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return float(inter / union) if union > 0 else 0.0


def suppress_obb(polys, confs, iou_th=0.5):
    """
    Simple greedy NMS on polygons using IoU.
    polys: (N,4,2), confs: (N,)
    """
    if len(polys) == 0:
        return []

    idxs = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
    keep = []

    while idxs:
        i = idxs.pop(0)
        keep.append(i)

        rest = []
        for j in idxs:
            if polygon_iou(polys[i], polys[j]) < iou_th:
                rest.append(j)
        idxs = rest

    return keep


# =========================
# DRAW + SAVE
# =========================
def draw_and_save_obb(r):
    """
    r: ultralytics.engine.results.Results
    """
    img = r.orig_img.copy()
    out_path = OUT_DIR / Path(r.path).with_suffix(".jpg").name

    # 0) 디버그: Ultralytics 기본 plot도 저장 (좌표 문제 / 모델 FP 구분)
    if SAVE_ULTRA_PLOT:
        try:
            dbg = r.plot()
            dbg_path = ULTRA_PLOT_DIR / Path(r.path).with_suffix(".jpg").name
            cv2.imwrite(str(dbg_path), dbg, [cv2.IMWRITE_JPEG_QUALITY, 90])
        except Exception:
            pass

    # 1) OBB 결과 없으면 저장만
    if r.obb is None or len(r.obb) == 0:
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return

    # 2) OBB polys/conf/cls (픽셀 좌표)
    polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
    confs = r.obb.conf.cpu().numpy()
    clss  = r.obb.cls.cpu().numpy().astype(int)

    # 3) confidence 1차 필터 (PRED와 동일 기준으로 통일)
    valid = confs >= CONF_TH_DRAW
    polys = polys[valid]
    confs = confs[valid]
    clss  = clss[valid]

    if len(polys) == 0:
        cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return

    # 4) 중복 제거 (custom polygon NMS)
    keep_idxs = suppress_obb(polys, confs, iou_th=OVERLAP_IOU)

    # 5) draw
    for i in keep_idxs:
        pts = polys[i].astype(int)
        cls_id = int(clss[i])
        conf = float(confs[i])

        if conf < CONF_TH_DRAW:
            continue
        if(obb_area(pts) < 20000):  # 너무 작은 OBB 무시
            cv2.polylines(img, [pts], True, (255, 255, 0), 4)
            continue
        
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        name = r.names.get(cls_id, str(cls_id))
        label = f"{name} {conf:.2f}"

        cv2.polylines(img, [pts], True, color, 4)

        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(
            img,
            label,
            (cx - 40, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


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
        conf=CONF_TH_PRED,
        iou=IOU_TH_PRED,
        max_det=MAX_DET,
        device=DEVICE,
        half=True,            # 문제 있으면 False로 바꿔서 테스트
        agnostic_nms=False,   # ✅ 디버그/안정성: class agnostic 끄기
        save=False,
        show=False,
        verbose=True,
    )
    t1 = time.perf_counter()
    print(f"Prediction time: {(t1 - t0):.3f} s")

    t0 = time.perf_counter()
    workers = max(1, cpu_count() // 4)
    with Pool(workers) as p:
        p.map(draw_and_save_obb, results)
    t1 = time.perf_counter()
    print(f"Save time: {(t1 - t0):.3f} s")

    print("Done.")
