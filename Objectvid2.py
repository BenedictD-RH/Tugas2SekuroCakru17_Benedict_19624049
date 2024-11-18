import cv2, sys, os, time
from random import randint

def boundRect(contours) :
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    return boundRect


tracker = cv2.TrackerKCF_create()

video = cv2.VideoCapture("Beyvid1.mp4")

ok, frame = video.read()

imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask1 = cv2.inRange(imghsv, (91,45,0), (102,255,255))
cv2.waitKey(0) 
contours, hierarchy = cv2.findContours(mask1, 1, 2)

contours_poly = [None]*len(contours)

ok = tracker.init(frame, boundRect(contours)[0])

while True:
    ok, frame = video.read()

    if not ok:
        break
    
    imghsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(imghsv, (91,58,0), (100,255,255))
    contours, hierarchy = cv2.findContours(mask1, 1, 2)
    
    contoursclean = []
    for c in contours:
        x = 1000
        if cv2.contourArea(c) > x:
            contoursclean.append(c) 
        
    if ok == True:
        xstart = 2000
        ystart = 2000
        xend = 0
        yend = 0
        detected = False
        for (x, y, width, height) in boundRect(contoursclean) :
            if x < xstart:
                xstart = x
                detected = True
            if y <ystart :
                ystart = y
                detected = True
            if x + height > xend :
                xend = x + height
                detected = True
            if y + width > yend :
                yend = y + width
                detected = True
        cv2.drawContours(frame, contoursclean, -1, (0,255,0), 5)
        if detected == True :
            cv2.rectangle(frame, (xstart - 50, ystart - 50), (xend + 50, yend + 50), (0,0,0), 5)

    cv2.putText(frame, 'Tracking', (100, 20), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255))

    frame = cv2.resize(frame, (800, 800))
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0XFF == 27:
        break