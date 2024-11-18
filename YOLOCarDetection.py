import cv2
from ultralytics import YOLO
yolo = YOLO('yolov8s.pt')


videoCap = cv2.VideoCapture("Highway2.mp4")
while True:
    k = 0
    ret, frame = videoCap.read()
    if not ret:
        continue
    results = yolo.track(frame, stream=True)
    
    for result in results:
        classes_names = result.names
    for box in result.boxes:
        if box.conf[0] > 0.2 and classes_names[int(box.cls[0])] == "car":
            [x1, y1, x2, y2] = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            class_name = classes_names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            k += 1
            
    cv2.putText(frame, f'{k} Cars detected', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    frame = cv2.resize(frame, (960, 540) )
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
videoCap.release()
cv2.destroyAllWindows()