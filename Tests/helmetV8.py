from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
from obj_detection import objectDetection
import cv2


detect = objectDetection()

source = "../RESOURCES\helmet.mp4"
model = YOLO("../models\hemletYoloV8_1.pt")
results = model.predict(source=source, show=True, conf=0.5)

cap = cv2.VideoCapture(source)

while True:
    ret,frame = cap.read()
    name, box_coord, obj_detected_frame = detect.detect_objects(frame) # getting the bounding box

    print(name)