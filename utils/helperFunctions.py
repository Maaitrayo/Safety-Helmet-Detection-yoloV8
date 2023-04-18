import cv2
import os
import csv


def show_file_size(file):
    file_size = os.path.getsize(file)
    file_size_mb = round(file_size / 1024, 2)

    print("File size is " + str(file_size_mb) + "MB")


def imageLoader(folder_path):
    items = os.listdir(folder_path)
    print(f"[!] Found {len(items)} images [!]")

    images_path_list = []
    for image in items:
        item_path = os.path.join(folder_path, image)
        images_path_list.append(item_path)

    return (images_path_list, items)


def saveResultCSV(result, output_folder_name):
    csv_path = os.path.join(output_folder_name, output_folder_name + ".csv")
    with open(csv_path, "w") as f1:
        writer = csv.writer(f1, delimiter=",")  # lineterminator='\n',
        writer.writerow(["Image Name", "Image Location", "Status"])
        for i in range(len(result)):
            row = result[i]
            writer.writerow(row)


def checkHeads(
    labels,
    image_name_list,
    image_path_list,
    image,
    csv_result_msg_final,
    i,
    output_folder_name,
):
    if "head" in labels:
        print("head found")
        image_name = f"{image_name_list[i]}"
        image_loc = os.path.join(f"{output_folder_name}/", image_name)
        # cv2.imwrite(image_loc, image, [cv2.IMWRITE_JPEG_QUALITY, 1])
        cv2.imwrite(image_loc, image)
        print("After Compression")
        show_file_size(image_loc)

        img_loc = image_path_list[i]
        message = "No Helmet"

        csv_result_msg_final.append([image_name, img_loc, message])

    return csv_result_msg_final
