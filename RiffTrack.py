import cv2
import time
from modules import HandTrackingModule as ht
from modules.GameModule import GameModule, findSongPathing

audio = findSongPathing("./songs")
game = GameModule(audio)

cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionConfidence=0.65, trackingConfidence=0.5)

volumes = [0, 0, 0, 0]
pTime = 0

while True:
    pVolume = volumes.copy()
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    volumes = detector.detectVolume(img, lmList, volumes=volumes)

    game.changeVolume(volumes)

    if volumes != pVolume:
        print(volumes)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(10)

    if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1:
        break

game.quit()
cap.release()
cv2.destroyAllWindows()
