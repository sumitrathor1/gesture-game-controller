import cv2
import os

label = "open"  # change: open, fist, one, two
count = 0

save_path = f"dataset/{label}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        cv2.imwrite(f"{save_path}/{count}.jpg", frame)
        print(f"Saved {count}")
        count += 1

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()