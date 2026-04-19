import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time

with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

engine = pyttsx3.init()
engine.setProperty("rate", 150)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

last_spoken = ""
last_spoken_time = 0
speak_delay = 2  # seconds between repeated speech

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read frame")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            sample = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(sample)[0]

    cv2.putText(
        frame,
        f"Prediction: {prediction}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    current_time = time.time()

    if prediction != "" and (
        prediction != last_spoken or current_time - last_spoken_time > speak_delay
    ):
        engine.say(str(prediction))
        engine.runAndWait()
        last_spoken = prediction
        last_spoken_time = current_time

    cv2.imshow("ASL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()