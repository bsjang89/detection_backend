from ultralytics import YOLO

def main():
    model = YOLO("yolov8m-obb.pt")
    model.train(
        data=r"D:\Datasets\Otoki\1\data.yaml",
        epochs=300,              # 50 → 80
        imgsz=768,              # ⭐ 640 → 768 (최소)
        device=0,
        cache="ram",
        lr0=0.005,             # ⭐ 핵심
        mosaic=0.5,
        close_mosaic=10,        # ⭐ 마지막 10epoch mosaic OFF
        mixup=0.0,
        workers=4,
        batch=32
    )

# def main():
#     model = YOLO("yolo26n-obb.pt")
#     model.train(
#         data=r"D:\Datasets\Shoppee\Training\data.yaml",  # 본인 경로로
#         epochs=50,
#         imgsz=640,
#         device=0,
#         cache="ram",
#         lr0=0.005,
#         mosaic=0.5,
#         mixup=0.0,
#         close_mosaic=0,
#         workers=4,   # <-- 아래 2번과 동일, 안전하게 같이 넣기        
#         batch=16
#     )

if __name__ == "__main__":
    main()


#.venv\Scripts\activate