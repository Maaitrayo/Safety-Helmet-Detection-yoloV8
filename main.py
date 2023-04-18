import cv2
import supervision as sv
import os
from datetime import datetime
import sys

from utils.helperFunctions import *
from ultralytics import YOLO


source = "../RESOURCES\helmet.mp4"
model = YOLO("../models\hemletYoloV8_100epochs.pt")

frame_wid = 640
frame_hyt = 480


def processImages(image_path_list, image_name_list, output_folder_name):
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=1)

    csv_result_msg_final = []

    for i in range(len(image_path_list)):
        frame = cv2.imread(image_path_list[i])
        print("Before Compression")
        show_file_size(image_path_list[i])

        image = cv2.resize(frame, (frame_wid, frame_hyt))

        results = model(image)[0]
        detections = sv.Detections.from_yolov8(results)
        labels = [f"{model.model.names[class_id]}" for _, _, class_id, _ in detections]

        image = box_annotator.annotate(
            scene=image,
            detections=detections
            # labels=labels
        )
        csv_result_msg_final = checkHeads(
            labels,
            image_name_list,
            image_path_list,
            image,
            csv_result_msg_final,
            i,
            output_folder_name,
        )

        cv2.imshow("Helmet Detection", image)
        if cv2.waitKey(1) == 27:
            break

    return csv_result_msg_final


if __name__ == "__main__":
    try:
        output_folder_name = datetime.now().strftime("%Y-%m-%d-%H_%M")
        os.makedirs(output_folder_name)
    except:
        print("[!] folder already exists [!]")
    folder_path = sys.argv[1]
    image_path_list, image_name_list = imageLoader(folder_path)
    result = processImages(image_path_list, image_name_list, output_folder_name)
    saveResultCSV(result, output_folder_name)
