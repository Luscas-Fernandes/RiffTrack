import cv2
import mediapipe as mp


class HandDetector:
    def __init__(
            self,
            mode=False,
            maxHands=4,
            detectionConfidence=0.5,
            trackingConfidence=0.5,
    ):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackingConfidence = trackingConfidence

        self.mpHands = mp.solutions.hands

        # default parameters
        self.hands = self.mpHands.Hands(
            self.mode,
            self.maxHands,
            1,
            self.detectionConfidence,
            self.trackingConfidence
        )

        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):

        AllLmLists = []

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                lmList = []
                for id, lmk in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lmk.x * w), int(lmk.y * h)
                    lmList.append([id, cx, cy])

                    if draw and id == 8:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255),
                                   cv2.FILLED)

                AllLmLists.append(lmList)

        return AllLmLists

    def clamp(self, x, lo, hi):
        return lo if x < lo else hi if x > hi else x

    def linearMapping(self, val, in_lo, in_hi, out_lo, out_hi):
        if in_hi == in_lo:
            return out_lo
        t = (val - in_lo) / (in_hi - in_lo)
        return out_lo + t * (out_hi - out_lo)

    def detectVolume(self, img, handsLmLists, y_top_margin=50, y_bottom_margin=50, volumes=None):
        if volumes is None:
            volumes = [0, 0, 0, 0]

        if not handsLmLists:
            return [0, 0, 0, 0]

        h, w, _ = img.shape
        y_top = y_top_margin
        y_bottom = h - y_bottom_margin

        last_hand = 0
        # handsLmLists = [mao0, mao1, mao2, mao3]
        for hand_idx, hand in enumerate(handsLmLists[:4]):  # no máximo 4 mãos
            if len(hand) <= 8:
                continue

            # hand[8] = [id, cx, cy]
            landmark_8 = hand[8]
            if not isinstance(landmark_8, (list, tuple)) or len(landmark_8) < 3:
                continue

            y = landmark_8[2]

            y = self.clamp(y, y_top, y_bottom)

            vol = int(round(self.linearMapping(y, y_bottom, y_top, 0, 100)))
            vol = self.clamp(vol, 0, 100)

            # Atualiza volume de forma suavizada
            volumes[hand_idx] = 0.7 * volumes[hand_idx] + 0.3 * vol

            last_hand = hand_idx

        for idx in range(last_hand + 1, len(volumes)):
            volumes[idx] = 0.0

        return [int(round(v)) for v in volumes]
