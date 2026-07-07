import cv2
import cvzone
from ultralytics import YOLO
import numpy as np
from sort import *

cap = cv2.VideoCapture("Road-Videos/Video1.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

model = YOLO("../Yolo-Weights/yolov8n.pt")
mask = cv2.imread("mask.png")

tracker = Sort(max_age=40, min_hits=3, iou_threshold=0.3)

vehcount = []
limits = [140, 500, 590, 500]

while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.resize(img, (1280, 720))
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, show=False, conf=0.5, verbose=False)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])
            clsName = classNames[cls]
            conf = float(box.conf[0])

            if clsName in ["car", "motorbike", "truck", "bicycle"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:

        x1, y1, x2, y2, _id = result
        x1, y1, x2, y2, _id = int(x1), int(y1), int(x2), int(y2), int(_id)

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w // 2
        cy = y1 + h // 2

        cvzone.cornerRect(img, (x1, y1, w, h), l=10, rt=2)
        cvzone.putTextRect(img, f"{_id}", (max(0, x1), max(35, y1)), scale=1, thickness=1, offset=3)

        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and abs(cy - limits[1]) <= 25:
            if _id not in vehcount:
                vehcount.append(_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, f"Count : {len(vehcount)}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 2, 2), 3)

    cv2.imshow("Vehicle Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
