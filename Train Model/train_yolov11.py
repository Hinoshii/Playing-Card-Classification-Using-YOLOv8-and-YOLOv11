import ultralytics
from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    results = model.train(data="yameaw.yaml", epochs=10, imgsz=640)

if __name__ == '__main__':
    main()
