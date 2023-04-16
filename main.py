import cv2
import supervision as sv
import os
import csv

from ultralytics import YOLO



source = "../RESOURCES\helmet.mp4"
model = YOLO("../models\hemletYoloV8_100epochs.pt")

frame_wid = 640
frame_hyt = 480

cap = cv2.VideoCapture(0)

def imageLoader(folder_path):
    items = os.listdir(folder_path)
    print(f"[!] Found {len(items)} images [!]")

    images_path_list = []
    for image in items:
        item_path = os.path.join(folder_path, image)
        images_path_list.append(item_path)
    # print(item_path)
    return(images_path_list, items)

def saveResultCSV(result):
    with open('result.csv','w') as f1:
        writer=csv.writer(f1, delimiter=',')#lineterminator='\n',
        for i in range(len(result)):
            row = result[i]
            writer.writerow(row)

def processImages(image_path_list, image_name_list):
    # if not cap.isOpened():
    #     print("Cannot open file")
    #     exit()

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=1
    )
    # count = 0
    # while True:
    csv_result_msg_final = []

    for i in range(len(image_path_list)):
        frame = cv2.imread(image_path_list[i])

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
            image_name = f"{image_name_list[i]}"
            # image_loc = "records/"+image_name
            image_loc = os.path.join("records/",image_name)
            cv2.imwrite(image_loc, image)

            csv_result_msg_row = []
            img_loc = image_path_list[i]
            message = "No Helmet"
            # csv_result_msg_row.append(img_loc)
            # csv_result_msg_row.append(message)
            csv_result_msg_final.append([img_loc, message])
            # csv_result_msg_row.clear()

        cv2.imshow("Helmet Detection", image)

        if (cv2.waitKey(0) == 27):
            break

    return csv_result_msg_final

if __name__ == "__main__":
    folder_path = input("Enter the folder path : ")
    # main()
    image_path_list, image_name_list = imageLoader(folder_path)
    result = processImages(image_path_list, image_name_list)
    saveResultCSV(result)

