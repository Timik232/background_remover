# write code for segmentation using ultralitytics
import ultralytics
from ultralytics import YOLO
ultralytics.checks()


def segmentation():

    # Load the model
    model = YOLO('yolov8s-seg.pt')

    # Train the model
    model.train(data='conf.yaml', epochs=100, imgsz=640, augment=True)


if __name__ == '__main__':
    segmentation()
