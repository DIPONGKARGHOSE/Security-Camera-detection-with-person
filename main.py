import cv2
import matplotlib.pyplot as plt
import winsound
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model, config_file)
classlabels = []
file_name = 'Labels.txt'
with open(file_name, 'rt') as fqt:
    classLabels = fqt.read().rstrip('\n').split('\n')
print(classLabels)
print(len(classLabels))
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot  open camera")
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
while True:
    ret, frame = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.55)
    print(ClassIndex)
    if (len(ClassIndex) != 0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
            if (ClassInd ==1):
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                cv2.putText(frame, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)
                winsound.PlaySound('alert.wav', winsound.SND_ASYNC)

    cv2.imshow('oSecurity Camera', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break;
cap.release()
cv2.destroyAllWindows()
