import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

label = input("Enter sign label (example A): ")

if not os.path.exists("dataset"):
    os.makedirs("dataset")

data = []

print("Press S to save sample")
print("Press Q to quit")

while True:

    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []

            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)
                landmarks.append(lm.z)

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s') and results.multi_hand_landmarks:
        data.append(landmarks)
        print("Sample saved:", len(data))

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

np.save(f"dataset/{label}.npy", data)

print("Dataset saved!")