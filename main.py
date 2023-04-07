import cv2
import supervision as sv

from ultralytics import YOLO



source = "../RESOURCES\helmet.mp4"
model = YOLO("../models\hemletYoloV8_100epochs.pt")

frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)

def main():
    if not cap.isOpened():
        print("Cannot open file")
        exit()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=1
    )
    count = 0
    while True:
        # ret, frame = cap.read()
        # frame = cv2.imread("D:\ANTEYE INTERNSHIP\HELMET DETECTION\DATASETS\DATASET LARGE\images\hard_hat_workers0.png") 
        frame = cv2.imread("D:\ANTEYE INTERNSHIP\HELMET DETECTION\RESOURCES\img2.jpeg")

        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break

        image = cv2.resize(frame, (frame_wid, frame_hyt))
        
        results = model(image)[0]
        detections = sv.Detections.from_yolov8(results)
        labels = [
            f"{model.model.names[class_id]}"
            for _, _, class_id, _ in detections
        ]

        image = box_annotator.annotate(
            scene=image, 
            detections=detections
            # labels=labels
        )
        if "head" in labels:
            print("head found")
            image_name = f"{count}.png"
            image_loc = "records/"+image_name
            cv2.imwrite(image_loc, image)

        cv2.imshow("Helmet Detection", image)

        if (cv2.waitKey(1) == 27):
            break

if __name__ == "__main__":
    main()

