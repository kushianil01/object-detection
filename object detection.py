import cv2
import numpy as np

#importing the files yolov4.cfg,yolov4.weights and coco.names.
config_path = r"C:\Users\Sushma\Downloads\yolov4.cfg"
weights_path = r"C:\Users\Sushma\Downloads\yolov4.weights"
labels_path = r"C:\Users\Sushma\Downloads\coco.names"

#DNN-deep neural network
#config_path=path to darknet configuration file
#weights_path=path to pre trained weights file.
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

#reading the contents of coco.names line by line
with open(labels_path, "r") as f:
    labels = [line.strip() for line in f.readlines()]

vid = cv2.VideoCapture(0) #used to capture the video, 0 is the parameter used, 1 and 2 can be used for high defintion cameras

while True:
    ret, frame = vid.read()
    #checks if the video is captured.
    if not ret or frame is None:
        print("Failed to capture image. Check the camera index or connection.")
        break

    height, width = frame.shape[:2]
    #this performs scaling, mean subtraction and optional channel swapping
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    layers = net.getLayerNames()#this obtains all the layernames present in the neural network i.e. net
    #this makes sure that it only includes the layer names that are included in the output image only.
    output_layers = [layers[i - 1] for i in net.getUnconnectedOutLayers()]

    detections = net.forward(output_layers)#these consist of the output layernames that are being detected

    boxes, confidences, class_ids = [], [], []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)#determines the class of the highest score 
            confidence = scores[class_id]
            if confidence > 0.5:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)#used to reduce the overlapping of boxes

    labels_detected = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(labels[class_ids[i]])
            labels_detected.append(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): #press q to exit from the application.
        break

vid.release()
cv2.destroyAllWindows()

if labels_detected:
    stringh = [f"I found a {labels_detected[0]}"]
    for label in labels_detected[1:]:
        stringh.append(f"a {label}")

    print(", ".join(stringh))
else:
    print("No objects detected.")

