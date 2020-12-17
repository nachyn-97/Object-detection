import cv2
import h as h
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layers_names = net.getLayerNames()
output_layers = [layers_names[i[0]-1]for i in net.getUnconnectedOutLayers()]

img = cv2.imread("room_ser.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0,), True, crop=False)


net.setInput(blob)
outs = net.forward(output_layers)



class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)

            x = int(center_x - w/2)
            y = int(center_y - height/2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

number_objects_detected = len(boxes)
for i in range(len(boxes)):
    x, y, w, h = boxes[i]
    label = classes[class_ids[i]]
    print(label)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
