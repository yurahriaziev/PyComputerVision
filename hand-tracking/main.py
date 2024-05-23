import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
# adjust the look of landmarks
landmark_spec = mpDraw.DrawingSpec(color=(0,0,255), thickness=4, circle_radius=5)
# adjust the look of connections between the landmarks
connection_spec = mpDraw.DrawingSpec(color=(255,255,255), thickness=3)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS, landmark_spec, connection_spec)

    cv2.imshow("Image", img)
    cv2.waitKey(1)