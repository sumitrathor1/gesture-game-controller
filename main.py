import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import threading
from tkinter import *
from tkinter import ttk

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Game mapping
game_controls = {
    "Hill Climb": {"acc": "right", "brake": "left"},
    "Subway Surfers": {"acc": "up", "brake": "down", "left": "left", "right": "right"},
    "Car Racing": {"acc": "w", "brake": "s", "left": "a", "right": "d"}
}

running = False

def release_keys():
    for key in ["up", "down", "left", "right", "w", "a", "s", "d"]:
        pyautogui.keyUp(key)

def control_game(action):
    release_keys()
    game = selected_game.get()
    controls = game_controls[game]

    if action == "open":
        pyautogui.keyDown(controls["acc"])
        status_label.config(text="Accelerating")

    elif action == "fist":
        pyautogui.keyDown(controls["brake"])
        status_label.config(text="Braking")

    elif action == "one" and "left" in controls:
        pyautogui.keyDown(controls["left"])
        status_label.config(text="Left")

    elif action == "two" and "right" in controls:
        pyautogui.keyDown(controls["right"])
        status_label.config(text="Right")

    else:
        status_label.config(text="Idle")

def start_camera():
    global running
    running = True
    threading.Thread(target=run_camera).start()

def stop_camera():
    global running
    running = False
    release_keys()

def run_camera():
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.append(lm.x)
                    landmark_list.append(lm.y)

                prediction = model.predict([landmark_list])[0]
                control_game(prediction)

        else:
            release_keys()
            status_label.config(text="No Hand")

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# UI
root = Tk()
root.title("Gesture Game Controller")
root.geometry("400x300")
root.configure(bg="#1e1e1e")

games = ["Hill Climb", "Subway Surfers", "Car Racing"]
selected_game = StringVar()
selected_game.set(games[0])

dropdown = ttk.Combobox(root, textvariable=selected_game, values=games)
dropdown.pack(pady=20)

Button(root, text="Start", command=start_camera, bg="green", fg="white").pack(pady=10)
Button(root, text="Stop", command=stop_camera, bg="red", fg="white").pack(pady=10)

status_label = Label(root, text="Idle", bg="#1e1e1e", fg="white", font=("Arial", 14))
status_label.pack(pady=20)

root.mainloop() 