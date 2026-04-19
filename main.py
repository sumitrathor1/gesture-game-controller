import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import threading
from collections import Counter, deque
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
active_key = None

prediction_window = deque(maxlen=7)
stable_prediction_min_count = 4
current_action = None
no_hand_frames = 0
no_hand_release_threshold = 6


def release_active_key():
    global active_key
    if active_key is not None:
        pyautogui.keyUp(active_key)
        active_key = None

def release_keys():
    global active_key
    for key in ["up", "down", "left", "right", "w", "a", "s", "d"]:
        pyautogui.keyUp(key)
    active_key = None


def get_smoothed_action(prediction):
    prediction_window.append(prediction)
    action, count = Counter(prediction_window).most_common(1)[0]

    if count >= stable_prediction_min_count:
        return action

    return None


def map_action_to_key_and_status(action, controls):
    if action == "open":
        return controls["acc"], "Accelerating"

    if action == "fist":
        return controls["brake"], "Braking"

    if action == "one" and "left" in controls:
        return controls["left"], "Left"

    if action == "two" and "right" in controls:
        return controls["right"], "Right"

    return None, "Idle"

def control_game(action):
    global active_key, current_action
    game = selected_game.get()
    controls = game_controls[game]

    next_key, next_status = map_action_to_key_and_status(action, controls)

    if next_key != active_key:
        release_active_key()
        if next_key is not None:
            pyautogui.keyDown(next_key)
            active_key = next_key

    if action != current_action:
        status_label.config(text=next_status)
        current_action = action

def start_camera():
    global running, no_hand_frames, current_action
    running = True
    no_hand_frames = 0
    current_action = None
    prediction_window.clear()
    threading.Thread(target=run_camera).start()

def stop_camera():
    global running
    running = False
    release_keys()

def run_camera():
    global no_hand_frames, current_action
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
                smoothed_action = get_smoothed_action(prediction)
                no_hand_frames = 0

                if smoothed_action is not None:
                    control_game(smoothed_action)

        else:
            no_hand_frames += 1
            if no_hand_frames >= no_hand_release_threshold:
                release_active_key()
                if current_action != "no_hand":
                    status_label.config(text="No Hand")
                    current_action = "no_hand"

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) == 27:
            break

    release_keys()
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