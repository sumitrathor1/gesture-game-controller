import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

data = []
labels = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                landmark_list = []

                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                data.append(landmark_list)
                labels.append(label)

print("Training...")

model = RandomForestClassifier()
model.fit(data, labels)

os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))

print("Model Saved!")