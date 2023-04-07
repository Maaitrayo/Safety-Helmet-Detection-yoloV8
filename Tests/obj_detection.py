import cv2
import torch
import numpy as np


class objectDetection:
    def __init__(self):
        ''' Downloading the model and initializing the parameters
            ARGS         :   NONE

            RETURNS      :   NONE
           
            PARAMETERS   :   device  -> Detects the computing device, GPU/CPU
                         :   model   -> Loads the pretrained yolov5s from torch 
                         :   classes -> Used to extract the name of the detected objects
        '''

        # self.model      =  torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model      =  torch.load("D:\ANTEYE INTERNSHIP\HELMET DETECTION\models\hemletYoloV8_1.pt")
        # self.classes    =  self.model.names
        self.device     =  'cuda' if torch.cuda.is_available() else 'cpu'

        print("... Loading Object Detection ...")
        print("USING DEVICE : ", self.device)
    

    def detect_objects(self, frame):
        ''' The main function where the object detection is done from a single frame(image) it calls another function plot_bounding_box for drawing the boxes
            ARGS        :    frame              -> input image frame

            RETURNS     :   name_lables        -> a list contating name of the detected objects in order
                        :   coordinates        -> position of the bounding boxes which have to be scaled (X1 = coordinate[i][0]*image width ) OR 
                        :   boox_coord         -> position of the bounding boxes as (x1,y1,x2,y2) 
                        :   obj_detected_frame -> image frame with bounding boxes along the detected objects, an output of plot_bounding_box

            PARAMETERS  :   frame_copy         -> copy of the original/raw image input frame 
                        :   name_labels        -> List to store the name of the detected objects
                        :   results            -> Stores the info about the detections from the raw image frame
                        :   labels             -> Some value correspomding to detected object name classification
                        :   coordinates        -> coordinates corresponding to the detected objects
                        :   name               -> string name of corresponding labels 
        '''
        frame_copy                      =  frame
        name_labels                     =  []
        self.model.to(self.device) 
        frame                           =  [frame]
        results                         =  self.model(frame)
        labels, coordinates             =  results.xyxyn[0][:,-1], results.xyxyn[0][:,:-1]

        obj_detected_frame, box_coord   =  self.plot_bounding_box(labels, coordinates, frame_copy)

        for i in range(len(labels)):
            name    =  self.object_classifier(labels[i])
            name_labels.append(name)

        print("object detection successful")
        return name_labels, box_coord, obj_detected_frame


    def object_classifier(self, pos):
        '''This function converts the labels to its corresponding object name like(car/bus/person...)
           ARGS         :    pos             -> A tensor ' tensor(7., device='cuda:0') '

           RETURNS      :    object_name     -> A string which contains the name of the detected object
           
           PARAMETERS   :   object_name      -> A string
        '''
        object_name     =  self.classes[int(pos)]
        return object_name


    def plot_bounding_box(self, labels, coordinates, frame):
        '''This function is used to plot the bounding boxes around the detected objects and put the object name above those boxes 
            after receiving the copy of the original frame as input. And calls object_classifire function to fetching object name
            ARGS        :   frame           -> input image frame
                        :   coordinates     -> coordinates corresponding to the detected objects
                        :   labels          -> Some value correspomding to detected object name classification

            RETURNS     :   frame           -> An image frame containing bounding boxes and text around the detected objects

            PARAMETERS  :   GREEN           -> colour green (0,255,0)
                        :   n               -> length of the variable labels
                        :   x1,x2,y1,y2     -> Positions of the bounding box
                        :   frame_x         -> width of the input image frame 
                        :   frame_y         -> heigth of the input image frame
                        :   row             -> first row of the coordinates, values of the first detected object in a single frame
        '''
        n                   =  len(labels)
        frame_x, frame_y    =  frame.shape[1], frame.shape[0]

        box_coord           =  []
        
        for i in range(n):
            row         =   coordinates[i]
            pos_ini     =   []

            if row[4]   >=  0.2:
                x1, y1, x2, y2  =  int(row[0]*frame_x), int(row[1]*frame_y), int(row[2]*frame_x), int(row[3]*frame_y)
                pos_ini         =  [x1, y1, x2, y2]
                GREEN           =  (0, 255, 0)
                box_coord.append(pos_ini)

                # cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
                # cv2.putText(frame, self.object_classifier(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)

        return frame, box_coord