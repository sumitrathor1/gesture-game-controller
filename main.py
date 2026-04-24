import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import pickle
import sklearn
import threading
import time
import os
import sys
from collections import Counter, deque
from tkinter import *
from tkinter import ttk

def resource_path(relative_path):
    """Return absolute path for local/dev run and PyInstaller onefile mode."""
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


# Load model
with open(resource_path("model/model.pkl"), "rb") as model_file:
    model = pickle.load(model_file)

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
current_game = "Hill Climb"

prediction_window = deque(maxlen=7)
stable_prediction_min_count = 4
current_action = None
no_hand_frames = 0
no_hand_release_threshold = 6
last_action_change_time = 0.0
car_racing_switch_delay_sec = 0.35
current_status = "Idle"
last_rendered_status = None


def handle_failsafe():
    global running, current_status, current_action
    running = False
    current_action = None
    current_status = "Mouse at screen corner. Move mouse and press Start again."


def safe_press(key):
    try:
        pyautogui.press(key)
        return True
    except pyautogui.FailSafeException:
        handle_failsafe()
        return False


def safe_key_down(key):
    try:
        pyautogui.keyDown(key)
        return True
    except pyautogui.FailSafeException:
        handle_failsafe()
        return False


def safe_key_up(key):
    try:
        pyautogui.keyUp(key)
    except pyautogui.FailSafeException:
        pass


def release_active_key():
    global active_key
    if active_key is not None:
        safe_key_up(active_key)
        active_key = None

def release_keys():
    global active_key
    for key in ["up", "down", "left", "right", "w", "a", "s", "d"]:
        safe_key_up(key)
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
    global active_key, current_action, last_action_change_time, current_status
    game = current_game
    controls = game_controls[game]

    if game == "Subway Surfers":
        # Trigger only once per gesture change for lane-switch/jump/slide.
        if action == current_action:
            return

        release_active_key()

        if action == "open":
            if not safe_press(controls["acc"]):
                return
            current_status = "Jump"
            current_action = action
            return

        if action == "fist":
            if not safe_press(controls["brake"]):
                return
            current_status = "Slide"
            current_action = action
            return

        if action == "one" and "left" in controls:
            if not safe_press(controls["left"]):
                return
            current_status = "Left"
            current_action = action
            return

        if action == "two" and "right" in controls:
            if not safe_press(controls["right"]):
                return
            current_status = "Right"
            current_action = action
            return

        current_status = "Idle"
        current_action = action
        return

    next_key, next_status = map_action_to_key_and_status(action, controls)

    if game == "Car Racing" and action != current_action:
        now = time.monotonic()
        if now - last_action_change_time < car_racing_switch_delay_sec:
            return
        last_action_change_time = now

    if next_key != active_key:
        release_active_key()
        if next_key is not None:
            if not safe_key_down(next_key):
                return
            active_key = next_key

    if action != current_action:
        current_status = next_status
        current_action = action

def start_camera():
    global running, no_hand_frames, current_action, last_action_change_time, current_status
    if running:
        current_status = "Camera already running"
        return

    running = True
    no_hand_frames = 0
    current_action = None
    last_action_change_time = 0.0
    current_status = "Starting camera..."
    prediction_window.clear()
    threading.Thread(target=run_camera, daemon=True).start()

def stop_camera():
    global running, current_status
    running = False
    release_keys()
    current_status = "Idle"


def on_game_change(event=None):
    global current_game, current_action, current_status
    current_game = selected_game.get()
    current_action = None
    release_keys()
    current_status = "Idle"


def refresh_status_label():
    global last_rendered_status
    if current_status != last_rendered_status:
        status_label.config(text=current_status)
        last_rendered_status = current_status
    root.after(100, refresh_status_label)

def run_camera():
    global no_hand_frames, current_action, current_status
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
                    current_status = "No Hand"
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
dropdown.bind("<<ComboboxSelected>>", on_game_change)

Button(root, text="Start", command=start_camera, bg="green", fg="white").pack(pady=10)
Button(root, text="Stop", command=stop_camera, bg="red", fg="white").pack(pady=10)

status_label = Label(root, text="Idle", bg="#1e1e1e", fg="white", font=("Arial", 14))
status_label.pack(pady=20)

on_game_change()
refresh_status_label()

root.mainloop() 