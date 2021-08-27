import cv2
import time
import numpy as np
import mediapipe as mp
from google.protobuf.json_format import MessageToDict

from djitellopy import Tello as djiTello

##################################### GLOBAL VARIABLES #####################################
WIDTH  = -1 # DEFAULT 960
HEIGHT = -1 # DEFAULT 720
DONT_FLY = True
DISPLAY  = True

# Class to run YOLO Object Detection and Localization and is based on https://github.com/arunponnusamy/object-detection-opencv
class MachineVision():
    # Constructor
    def __init__(self, display):
        yoloPrePath = "models/Yolo/"
        self.yolo_filenames = {"class"  : yoloPrePath + "yolov3.txt",
                               "weight" : yoloPrePath + "yolov3.weights",
                               "config" : yoloPrePath + "yolov3.cfg"}

        fdPrePath = "models/CV_FaceDetection/"
        self.fd_filenames = {"model"  : fdPrePath + "res10_300x300_ssd_iter_140000.caffemodel",
                             "config" : fdPrePath + "fd.cfg"}

        # Parse Yolo class file for list of classes. Also create corresponding list of colors
        with open(self.yolo_filenames["class"], 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Import networks
        self.yolo_net = cv2.dnn.readNet(self.yolo_filenames["weight"], self.yolo_filenames["config"])
        self.fd_net   = cv2.dnn.readNetFromCaffe(self.fd_filenames["config"], self.fd_filenames["model"])

        # Import hand detection and tracking
        MAX_NUM_HANDS = 2
        MIN_DETECTION_CONFIDENCE = 0.5
        MIN_TRACKING_CONFIDENCE  = 0.5
        self.mpHands = mp.solutions.hands
        self.hands   = self.mpHands.Hands(False, MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE)
        self.mpDraw  = mp.solutions.drawing_utils

        # Class indexes for specific objects
        self.yolo_class_indexes = {"person" : self.classes.index("person"),
                                   "dog"    : self.classes.index("dog")}

        self.DISPLAY = display

    # Internal function for getting list of output_layers that need to be read from.
    # This is needed because YOLO architecture has multiple output layers
    def get_output_layers(self, net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # Internal function to draw a bounding box with class_id and confidence
    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2) # inputs top-left corner, and bottom-right corner
        cv2.putText(img, label + "(" + str(round(confidence, 3)) + ")", (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Function to return predictions on an inputted image
    def yolo_predict(self, image, confidence_thresh = 0.5):
        Height, Width = image.shape[:2]

        # Run image through pre-trained network
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.00392, size=(416,416), mean=(0,0,0), swapRB=True, crop=False)
        self.yolo_net.setInput(blob)
        outs = self.yolo_net.forward(self.get_output_layers(self.yolo_net))

        # Loop through Yolo network output(s) and sort into class_ids, confidences, boxes (x_top_left, y_top_left, width, height)
        boxes = []
        confidences = []
        class_ids = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_thresh:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Use Non-Max Suppression to remove redundant bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=confidence_thresh, nms_threshold=0.4) # Does non-max suppression where score must be >=0.5 and non-max suppresion calculation >=0.4
        nmsBoxes = []
        nmsConf = []
        nmsClassId = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            [x, y, w, h] = box

            nmsBoxes.append(box)
            nmsConf.append(confidences[i])
            nmsClassId.append(class_ids[i])
            if self.DISPLAY:
                self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        if self.DISPLAY:
            cv2.imshow("object detection", image)
            cv2.waitKey(1)

        return nmsBoxes, nmsConf, nmsClassId

    def hand_detection_and_tracking(self, image):
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if self.DISPLAY:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)

        # Pack landmark info into a list of of dictionaries defining locations of specific hand landmarks
        handNo = 0
        landmarkList = []
        if self.results.multi_hand_landmarks:
            numHands = len(self.results.multi_hand_landmarks)
            landmarkList = np.zeros((numHands, 21, 2))
            for handIdx, currHand in enumerate(self.results.multi_hand_landmarks):
                # Process landmarks for each hand
                for id, lm in enumerate(currHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h) # output is given in percentage [0, 1] - convert to pixels in our image

                    landmarkList[handIdx][id] = [cx, cy]

        # Parse handedness result and print for debug (if display is on)
        handednessList = []
        if self.results.multi_handedness:
            handednessList = [0] * numHands
            for idx, currHandedness in enumerate(self.results.multi_handedness):
                dict = MessageToDict(currHandedness)
                dict = dict['classification'][0]

                #index = dict['index']
                score = dict['score']
                label = "Right" if dict['label'] == "Left" else "Left" # flip handedness because we didn't flip input image

                handednessList[idx] = [score, label]

        if self.DISPLAY:
            # Add labelling of enum points to the output
            for idx, hand in enumerate(landmarkList):
                for lm_id, lm in enumerate(hand):
                    tag = str(self.mpHands.HandLandmark(lm_id))
                    landmarkName = tag.split(".")[-1]


                    color = (0, 0, 255) if handednessList[idx][1] == "Right" else (0, 255, 0)
                    cv2.putText(image, landmarkName, (int(lm[0]), int(lm[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
                    if landmarkName == "WRIST":
                        cv2.putText(image, handednessList[idx][1], (int(lm[0]), int(lm[1]+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("HAND detection", image)
            cv2.waitKey(1)
        return image, landmarkList, handednessList

    def facial_detection(self, image):
        h, w = image.shape[:2]

        # Run image through pre-trained network
        blob = cv2.dnn.blobFromImage(image=image, scalefactor=1.0, size=(300, 300), mean=(104.0, 117.0, 123.0))
        self.fd_net.setInput(blob)
        faces = self.fd_net.forward()

        #to draw faces on image
        boxes = []
        confidences = []
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                [x, y, x1, y1] = faces[0, 0, i, 3:7]*np.array([w, h, w, h]) # first item represents top left and bottom right corner ass x, y, x2, y2 in [0, 1]. Then multiply by h and width to get approx number of pixels from [0, 1] output
                box = [round(x), round(y), round(x1), round(y1)] # round to get bounding box in pixels

                if self.DISPLAY:
                    color = (0, 0, 255)
                    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(image, "face (" + str(round(confidence, 3)) + ")", (box[0]-10,box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if self.DISPLAY:
            cv2.imshow("facial recognition", image)
            cv2.waitKey(1)

        return boxes, confidences

class Tello(djiTello):
    def __init__(self, width, height, DONT_FLY):
        #self.drone = Tello()
        super().__init__()
        self.width = width
        self.height = height
        self.fly_enabled = not DONT_FLY

        # Connect to Tello
        self.connect()
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 0

        # Turn on camera stream
        self.streamon()

        print("Battery: " + str(self.get_battery()) + "%")

    # Gets an image from the drone
    def getImage(self):
        frame_read = self.get_frame_read()
        img = frame_read.frame

        # Use actual image widths and/or heights if not specified
        if self.width == -1:
            self.width = img.shape[1]
            print("Width not set. Default to camera resolution: " + str(self.width))
        if self.height == -1:
            self.height = img.shape[0]
            print("Height not set. Default to camera resolution: " + str(self.height))

        return cv2.resize(img, (self.width, self.height))

    # Motion command wrapper functions
    def takeoff(self):
        if self.fly_enabled:
            super().takeoff()
            time.sleep(3)
    def land(self):
        if self.fly_enabled:
            super().land()
            time.sleep(3)

def main():
    global WIDTH, HEIGHT, DONT_FLY

    # Connect to tello
    firstTick = True
    tello = Tello(WIDTH, HEIGHT, DONT_FLY)
    mv = MachineVision(DISPLAY)

    # Main control loop
    while True:
        # Takeoff on the first tick
        if firstTick:
            input("Ready to takeoff?")
            tello.takeoff()

            firstTick = False

        # Get and display image from drone
        img = tello.getImage()

        # Do Object Detection and Localization on the image
        boxes, confidences = mv.facial_detection(img.copy())
        boxes, confidences, class_ids = mv.yolo_predict(img.copy())

        # Do hand detection
        foundHands_img, landmarkList, handednessList = mv.hand_detection_and_tracking(img.copy())


        #cv2.imshow("MyResult", img)

        # WAIT FOR THE 'Q' BUTTON TO STOP
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    tello.land()
        #    break

if __name__ == "__main__":
    main()
