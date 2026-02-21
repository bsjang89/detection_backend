from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"D:\Datasets\Coupang\Trainset\data.yaml",  # 본인 경로로
        epochs=50,
        imgsz=640,
        device=0,
        cache="ram",
        lr0=0.005,
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=0,
        workers=4,   # <-- 아래 2번과 동일, 안전하게 같이 넣기        
        batch=16
    )

if __name__ == "__main__":
    main()