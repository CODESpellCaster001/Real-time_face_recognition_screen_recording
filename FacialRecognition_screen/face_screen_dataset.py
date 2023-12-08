import cv2
import os
import numpy as np
from PIL import ImageGrab

# Create dataset directory if it does not exist
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Load the pre-trained face detector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\nEnter user id and press <return>: ')

print("\n[INFO] Initializing face capture. Look at the screen and wait...")

# Initialize individual sampling face count
count = 0

while True:
    # Capture the screen
    screen = ImageGrab.grab(bbox=(0, 0, 1920, 1080))  # Adjust the bbox based on your screen resolution
    frame = np.array(screen)

    # Convert BGR to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite(f"{dataset_dir}/User.{str(face_id)}.{str(count)}.jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', frame)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 30:  # Take 30 face samples and stop
        break

# Do a bit of cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
