import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math
import time
from collections import deque

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model("Model/keras_model.h5")
labels = open("Model/labels.txt", "r").read().splitlines()

offset = 20
imgSize = 224  # Match model input shape
print(model.input_shape)

message = ""
movement_history = []
max_history = 15
wave_threshold = 60  # Adjust based on test

# For detecting Z
z_motion_history = deque(maxlen=15)
last_z_time = 0

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        cx, cy = hand['center']  # Get center for motion tracking

        # Check if all 5 fingers are up
        fingers = detector.fingersUp(hand)
        if fingers == [1, 1, 1, 1, 1]:  # All fingers up
            movement_history.append(cx)
            if len(movement_history) > max_history:
                movement_history.pop(0)

            # Detect waving (left-right motion)
            if len(movement_history) == max_history:
                delta = max(movement_history) - min(movement_history)
                if delta > wave_threshold:
                    message += "hello "
                    print("[Wave + 5 Fingers] Added: hello")
                    movement_history.clear()
        else:
            movement_history.clear()  # reset if hand shape is not full open

        # Z detection: index finger only
        if fingers == [1, 0, 0, 0, 0]:
            z_motion_history.append((cx, cy))
            if len(z_motion_history) == z_motion_history.maxlen:
                dx1 = z_motion_history[5][0] - z_motion_history[0][0]  # right
                dy1 = z_motion_history[5][1] - z_motion_history[0][1]

                dx2 = z_motion_history[10][0] - z_motion_history[5][0]  # down-left
                dy2 = z_motion_history[10][1] - z_motion_history[5][1]

                dx3 = z_motion_history[14][0] - z_motion_history[10][0]  # up-right
                dy3 = z_motion_history[14][1] - z_motion_history[10][1]

                if dx1 > 20 and dx2 < -15 and dy2 > 10 and dx3 > 15 and dy3 < -10:
                    if time.time() - last_z_time > 2:
                        message += "Z"
                        print("[Motion + Index Finger] Added: Z")
                        last_z_time = time.time()
        else:
            z_motion_history.clear()

        # Prepare image for classification
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # Normalize and predict
        img_input = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype('float32') / 255.0
        img_input = np.expand_dims(img_input, axis=0)

        prediction = model.predict(img_input)
        index = np.argmax(prediction)

        # Display prediction
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Show current message
        cv2.putText(imgOutput, message, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw hand center
        cv2.circle(imgOutput, (cx, cy), 8, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
