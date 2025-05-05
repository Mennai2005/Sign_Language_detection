import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
model = load_model("Model/sign_language_model.h5")
labels = open("Model/labels.txt", "r").read().splitlines()

offset = 20
imgSize = 224  # Changed from 300 to 256 (to match model input shape)
print(model.input_shape)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
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

        # Convert to RGB & normalize (if model expects [0,1] range)
        img_input = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        img_input = img_input.astype('float32') / 255.0  # Normalize
        img_input = np.expand_dims(img_input, axis=0)  # Add batch dimension (1, 256, 256, 3)

        # Predict
        prediction = model.predict(img_input)
        index = np.argmax(prediction)

        # Display prediction
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)
        
    cv2.imshow("Image", imgOutput)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()