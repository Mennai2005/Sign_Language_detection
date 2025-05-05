import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import math
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Sign Language Detection")
        self.window.geometry("1200x800")

        # Initialize variables
        self.current_message = []
        self.last_prediction = ""
        self.prediction_threshold = 0.8
        self.cooldown_time = 1  # Increase this value to reduce prediction frequency
        self.last_prediction_time = time.time()
        self.is_running = True

        # Initialize detection components
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=2)
        self.model = load_model("Model/sign_language_model.h5")
        # Modified to remove numbers from labels
        self.labels = [label.split(' ')[-1] for label in open("Model/labels.txt", "r").read().splitlines()]
        self.offset = 20
        self.imgSize = 224

        # Add timer variable
        self.detection_time = 0
        self.start_time = None
        
        # Add countdown variables
        self.countdown_active = False
        self.countdown_start = 0
        self.countdown_duration = 3  # 3 seconds countdown
        self.next_letter = None
        
        # Add suggestion variables
        self.suggestions = []
        self.current_word = ""
        
        # Create GUI elements
        self.create_gui()

        # Start video thread
        self.video_thread = threading.Thread(target=self.update_video, daemon=True)
        self.video_thread.start()

    def create_gui(self):
        # Create main frames
        self.left_frame = ttk.Frame(self.window)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = ttk.Frame(self.window)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Video display
        self.video_label = ttk.Label(self.left_frame)
        self.video_label.pack()

        # Add timer display
        self.timer_frame = ttk.LabelFrame(self.right_frame, text="Detection Time")
        self.timer_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.timer_label = ttk.Label(self.timer_frame, text="0.00 ms", font=('Arial', 12))
        self.timer_label.pack(padx=5, pady=5)

        # Message display
        self.message_frame = ttk.LabelFrame(self.right_frame, text="Detected Message")
        self.message_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.message_text = tk.Text(self.message_frame, height=10, width=40, font=('Arial', 12))
        self.message_text.pack(padx=5, pady=5)

        # Control buttons
        self.button_frame = ttk.Frame(self.right_frame)
        self.button_frame.pack(fill=tk.X, padx=5, pady=5)

        self.clear_button = ttk.Button(self.button_frame, text="Clear Message", command=self.clear_message)
        self.clear_button.pack(side=tk.LEFT, padx=5)

        self.quit_button = ttk.Button(self.button_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack(side=tk.RIGHT, padx=5)

        # Add countdown display
        self.countdown_frame = ttk.LabelFrame(self.right_frame, text="Next Sign")
        self.countdown_frame.pack(fill=tk.X, padx=5, pady=5, before=self.message_frame)
        
        self.countdown_label = ttk.Label(self.countdown_frame, text="Ready", font=('Arial', 16, 'bold'))
        self.countdown_label.pack(padx=5, pady=5)

        # Add next letter display
        self.next_letter_label = ttk.Label(self.countdown_frame, text="", font=('Arial', 14))
        self.next_letter_label.pack(padx=5, pady=5)

        # Add suggestions display
        self.suggestions_frame = ttk.LabelFrame(self.right_frame, text="Word Suggestions")
        self.suggestions_frame.pack(fill=tk.X, padx=5, pady=5, before=self.message_frame)
        
        self.suggestions_text = tk.Text(self.suggestions_frame, height=2, width=40, font=('Arial', 12))
        self.suggestions_text.pack(padx=5, pady=5)
        
        # Add suggestion buttons
        self.suggestion_buttons_frame = ttk.Frame(self.suggestions_frame)
        self.suggestion_buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.suggestion_buttons = []
        for i in range(3):  # Create 3 suggestion buttons
            btn = ttk.Button(self.suggestion_buttons_frame, text="", command=lambda x=i: self.use_suggestion(x))
            btn.pack(side=tk.LEFT, padx=5, expand=True)
            self.suggestion_buttons.append(btn)

    def update_video(self):
        while self.is_running:
            success, img = self.cap.read()
            if not success:
                continue

            imgOutput = img.copy()
            hands, img = self.detector.findHands(img)
            
            current_time = time.time()

            # Handle countdown
            if self.countdown_active:
                remaining = self.countdown_duration - (current_time - self.countdown_start)
                if remaining > 0:
                    self.countdown_label.configure(text=f"Get Ready: {int(remaining)}s")
                    # Skip detection during countdown
                    imgOutput = cv2.resize(imgOutput, (480, 360))
                    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                    photo = ImageTk.PhotoImage(image=Image.fromarray(imgOutput))
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                    continue
                else:
                    self.countdown_active = False
                    self.countdown_label.configure(text="Make Sign Now!")
                    self.last_prediction_time = current_time  # Reset prediction timer

            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                if (current_time - self.last_prediction_time) > self.cooldown_time:
                    imgWhite = np.ones((self.imgSize, self.imgSize, 3), np.uint8) * 255
                    try:
                        imgCrop = img[max(0, y - self.offset):min(img.shape[0], y + h + self.offset), 
                                     max(0, x - self.offset):min(img.shape[1], x + w + self.offset)]
                        
                        # Add contrast adjustment
                        imgCrop = cv2.convertScaleAbs(imgCrop, alpha=1.2, beta=0)
                        
                        # Add noise reduction
                        imgCrop = cv2.GaussianBlur(imgCrop, (3,3), 0)
                        aspectRatio = h / w
                        if aspectRatio > 1:
                            k = self.imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, self.imgSize))
                            wGap = math.ceil((self.imgSize - wCal) / 2)
                            imgWhite[:, wGap:wCal + wGap] = imgResize
                        else:
                            k = self.imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (self.imgSize, hCal))
                            hGap = math.ceil((self.imgSize - hCal) / 2)
                            imgWhite[hGap:hCal + hGap, :] = imgResize

                        img_input = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                        img_input = img_input.astype('float32') / 255.0
                        img_input = np.expand_dims(img_input, axis=0)

                        prediction = self.model.predict(img_input)
                        index = np.argmax(prediction)
                        confidence = prediction[0][index]

                        if confidence > self.prediction_threshold:
                            predicted_sign = self.labels[index]
                            if predicted_sign != self.last_prediction:
                                self.add_to_message(predicted_sign)
                                self.last_prediction = predicted_sign
                                self.last_prediction_time = current_time
                                
                                # Start countdown for next letter
                                self.countdown_active = True
                                self.countdown_start = current_time
                                # Set next letter (you can modify this logic)
                                next_index = (index + 1) % len(self.labels)
                                self.next_letter = self.labels[next_index]
                                self.next_letter_label.configure(text=f"Next Sign: {self.next_letter}")
                                self.countdown_label.configure(text=f"Get Ready: {self.countdown_duration}s")
                    except Exception as e:
                        print(f"Error during image processing: {e}")

                    # Always show the last prediction
                    cv2.putText(imgOutput, self.labels[index] if 'index' in locals() else "", 
                              (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-self.offset, y-self.offset), 
                                (x + w + self.offset, y + h + self.offset), (255, 0, 255), 4)

                # Reduce the resize resolution for better performance
                imgOutput = cv2.resize(imgOutput, (480, 360))  # Reduced from 640x480
                imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(imgOutput))
                
                self.video_label.configure(image=photo)
                self.video_label.image = photo

    def add_to_message(self, sign):
        if sign == "space":
            # When space is detected, add the current word to suggestions
            if self.current_word:
                self.update_suggestions(self.current_word)
            self.current_word = ""
            self.current_message.append(" ")
        elif sign == "delete":
            if self.current_message:
                self.current_message.pop()
                # Update current word when deleting
                message = ''.join(self.current_message)
                self.current_word = message.split()[-1] if message.split() else ""
        else:
            self.current_message.append(sign)
            self.current_word += sign
            # Update suggestions as user types
            self.update_suggestions(self.current_word)
        
        # Update message display
        message_text = ''.join(self.current_message)
        self.message_text.delete(1.0, tk.END)
        self.message_text.insert(tk.END, message_text)

    def update_suggestions(self, current_word):
        # Simple suggestion logic - you can enhance this
        suggestions = []
        if current_word:
            # Add some basic suggestions (you can implement more sophisticated logic)
            suggestions = [
                current_word + "ing",
                current_word + "ed",
                current_word + "s"
            ]
            
            # Update suggestion buttons
            for i, suggestion in enumerate(suggestions[:3]):
                self.suggestion_buttons[i].configure(text=suggestion)

    def use_suggestion(self, index):
        if index < len(self.suggestion_buttons):
            suggestion = self.suggestion_buttons[index]['text']
            if suggestion:
                # Remove current word and add suggestion
                while self.current_word and self.current_message:
                    self.current_message.pop()
                self.current_message.extend(list(suggestion))
                self.current_word = suggestion
                
                # Update display
                message_text = ''.join(self.current_message)
                self.message_text.delete(1.0, tk.END)
                self.message_text.insert(tk.END, message_text)

    def clear_message(self):
        self.current_message = []
        self.current_word = ""
        self.message_text.delete(1.0, tk.END)
        self.last_prediction = ""
        # Clear suggestions
        for btn in self.suggestion_buttons:
            btn.configure(text="")

    def quit_app(self):
        self.is_running = False
        self.window.quit()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()

    # Cleanup
    app.cap.release()
    cv2.destroyAllWindows()