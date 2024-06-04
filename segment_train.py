from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')
model.train(data=os.path.join('datasets', 'data.yaml'), epochs=100, imgsz=640, augment=True)