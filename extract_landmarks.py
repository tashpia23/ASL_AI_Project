import cv2
import mediapipe as mp
import numpy as np
from datasets import load_dataset

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

dataset = load_dataset("ZahidYasinMittha/American-Sign-Language-Dataset")

landmark_data = []

for sample in dataset["train"]:
    
    video_path = sample["video"]["path"]
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.append(lm.x)
                    landmarks.append(lm.y)
                    landmarks.append(lm.z)

                landmark_data.append(landmarks)

    cap.release()

np.save("asl_landmarks.npy", landmark_data)

print("Landmarks extracted and saved!")