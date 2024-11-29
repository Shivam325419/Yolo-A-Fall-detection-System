# from ultralytics import YOLO
# # from ultralytics.yolo.v8.detect.predict import Detection
# import cv2

# model = YOLO("yolov5-master/best.pt")
# model.predict(source="0", show=True, conf=0.4)

from ultralytics import YOLO
import cv2

# Load the YOLOv5 model
model = YOLO("best.pt")

# Perform real-time object detection on webcam feed (source="0")
model.predict(source="0", show=True, conf=0.4)

