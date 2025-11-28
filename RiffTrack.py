import cv2
import time
import HandTrackingModule as ht

cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionConfidence=0.7)

volumes = [0, 0, 0, 0]

pTime = 0

while True:
    pVolume = volumes.copy()
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    volumes = detector.changeVolume(img, lmList, volumes=volumes)

    if volumes != pVolume:
        print(volumes)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
