import cv2
import time
import HandTrackingModule as ht
import pygame
import random

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

songs = ['crawling', 'somwhereib']

sounds = [
    pygame.mixer.Sound("songs/somewhereib/other.wav"),
    pygame.mixer.Sound("songs/somewhereib/drums.wav"),
    pygame.mixer.Sound("songs/somewhereib/bass.wav"),
    pygame.mixer.Sound("songs/somewhereib/vocals.wav"),
]

channels = [
    pygame.mixer.Channel(0),
    pygame.mixer.Channel(1),
    pygame.mixer.Channel(2),
    pygame.mixer.Channel(3),
]

for i in range(4):
    channels[i].play(sounds[i], loops=-1)
    channels[i].set_volume(0.0)


cap = cv2.VideoCapture(0)
detector = ht.HandDetector(detectionConfidence=0.65, trackingConfidence=0.5)

volumes = [0, 0, 0, 0]

pTime = 0

while True:
    pVolume = volumes.copy()
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    volumes = detector.changeVolume(img, lmList, volumes=volumes)

    for i, vol in enumerate(volumes):
        # volumes[i] está entre 0 e 100 -> normalizar para 0.0–1.0
        vol_norm = max(0.0, min(1.0, vol / 100.0))
        channels[i].set_volume(vol_norm)

    if volumes != pVolume:
        print(volumes)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2,
                (0, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(10)
