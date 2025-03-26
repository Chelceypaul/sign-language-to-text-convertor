import cv2
import numpy as np
import os
import time
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# Define the gesture name (change for each new gesture)
gesture_name = "Thank you"

# Create dataset directories
dataset_path = f"dataset/{gesture_name}"
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Initialize detectors
hand_detector = HandDetector(maxHands=2)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start webcam
cap = cv2.VideoCapture(0)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect hands
    hands, img = hand_detector.findHands(img)

    # Detect body pose
    results = pose.process(img_rgb)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            hand_label = "right_hand" if hand["type"] == "Right" else "left_hand"

            # Crop and save hand image
            try:
                imgCrop = img[y-20:y+h+20, x-20:x+w+20]
                imgCrop = cv2.resize(imgCrop, (128, 128))
                filename = f"{dataset_path}/{hand_label}_{time.time()}.jpg"
                cv2.imwrite(filename, imgCrop)
                print(f"Saved {filename}")

            except:
                pass

    # Save full-body pose if detected
    if results.pose_landmarks:
        filename = f"{dataset_path}/full_body_{time.time()}.jpg"
        cv2.imwrite(filename, img)
        print(f"Saved {filename}")

    cv2.imshow("Dataset Collection", img)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
