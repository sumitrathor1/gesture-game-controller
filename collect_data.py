import cv2
import os
import time

label = "fist"  # change: open, fist, one, two
target_images = 100
capture_interval_sec = 3
break_every_images = 25
break_duration_sec = 30

save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

existing_indices = []
for file_name in os.listdir(save_path):
    name, ext = os.path.splitext(file_name)
    if ext.lower() == ".jpg" and name.isdigit():
        existing_indices.append(int(name))

if existing_indices:
    count = max(existing_indices) + 1
else:
    count = 0

session_count = 0

cap = cv2.VideoCapture(0)
last_capture_time = time.time() - capture_interval_sec

print(f"Auto capture started for label '{label}'")
print(f"Target images: {target_images}, Interval: {capture_interval_sec} seconds")
print(f"Starting index: {count}")
print(f"Break: every {break_every_images} images, pause {break_duration_sec} seconds")
print("Press ESC to stop early")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    current_time = time.time()
    if current_time - last_capture_time >= capture_interval_sec:
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        print(f"Saved {count}")
        count += 1
        session_count += 1
        last_capture_time = current_time

        if session_count % break_every_images == 0 and count < target_images:
            print(f"Break time: waiting {break_duration_sec} seconds...")
            time.sleep(break_duration_sec)
            print("Break over. Resuming capture.")

        if count >= target_images:
            print("Target reached. Stopping capture.")
            break

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()