import cv2
import torch
import numpy as np

path = "../helmetV1.pt"

model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

# cap = cv2.VideoCapture("../helmet/helmet.mp4")
# count = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
    
#     frame = cv2.resize(frame,(700,500))
#     results = model(frame)
#     framme = np.squeeze(results.render())

#     cv2.imshow("frame", frame)
#     cv2.waitKey(1)
frame = cv2.imread("../img2.jpeg")
frame = cv2.resize(frame,(700,500))
results = model(frame)
framme = np.squeeze(results.render())

cv2.imshow("frame", frame)
cv2.waitKey(0)
