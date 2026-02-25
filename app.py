import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import tkinter as tk
from PIL import Image, ImageTk
import threading
import time

# ================= GLOBAL STATE ======================
camera_running = False
gesture_locked = False
current_gesture = None
current_step = 0
last_detect_time = 0

# ================= LOAD TRAINED MODEL =================
with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

# ================= MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ================= VOICE ======================
def speak(text):
    def run():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("TTS Error:", e)
    threading.Thread(target=run, daemon=True).start()

# ================= GESTURE INSTRUCTIONS =======
gesture_actions = {
    "OPEN": [
        "Collect and fill account opening form",
        "Proceed to KYC verification desk",
        "Submit documents at Counter 2"
    ],
    "CLOSE": ["Proceed to account closure desk"],
    "BALANCE": ["Provide account number"],
    "WITHDRAW": ["Fill withdrawal slip", "Go to cash counter"],
    "DEPOSIT": ["Fill deposit slip", "Go to deposit counter"],
    "UPDATE": ["Update KYC details"],
    "HELP": ["Staff assistance will arrive"],
    "WAIT": ["Please wait for your token"],
    "DONE": ["Thank you. Process completed"]
}

# ================= GUI ========================
root = tk.Tk()
root.title("SIGNBANK – Gesture-Based Banking Assistant")
root.geometry("1200x700")
root.configure(bg="#F8FAFC")

title = tk.Label(root, text="SIGNBANK", font=("Arial", 28, "bold"),
                 fg="#1E3A8A", bg="#F8FAFC")
title.pack(pady=5)

subtitle = tk.Label(root, text="AI-Powered Sign Language Banking Assistant",
                    font=("Arial", 14), fg="#475569", bg="#F8FAFC")
subtitle.pack()

main_frame = tk.Frame(root, bg="#F8FAFC")
main_frame.pack(expand=True, fill="both", pady=10)

video_label = tk.Label(main_frame, bg="black")
video_label.pack(side="left", padx=15)

info_label = tk.Label(main_frame,
    text="Click START to begin gesture detection",
    font=("Arial", 18), fg="#0F172A", bg="#E5E7EB",
    wraplength=450, justify="left", padx=20, pady=20
)
info_label.pack(side="right", padx=15, fill="y")

btn_frame = tk.Frame(root, bg="#F8FAFC")
btn_frame.pack(pady=10)

# ================= CAMERA =====================
cap = cv2.VideoCapture(0)

# ================= UPDATE FRAME =================
def update_frame():
    global gesture_locked, current_gesture, current_step, last_detect_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)

    if camera_running:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if not gesture_locked and result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                landmarks = []
                for lm in hand.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                proba = model.predict_proba([landmarks])[0]
                confidence = max(proba)
                prediction = model.classes_[np.argmax(proba)]

                print("Pred:", prediction, "Conf:", confidence)

                if confidence < 0.60:
                    continue

                now = time.time()
                if now - last_detect_time < 1:
                    continue

                last_detect_time = now

                if prediction in gesture_actions:
                    current_gesture = prediction
                    current_step = 0
                    instruction = gesture_actions[prediction][0]

                    info_label.config(
                        text=f"Gesture: {prediction}\nStep 1\n\n{instruction}"
                    )

                    speak(instruction)
                    gesture_locked = True

    img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video_label.imgtk = img
    video_label.config(image=img)

    root.after(10, update_frame)

# ================= BUTTON FUNCTIONS =============
def start_detection():
    global camera_running
    camera_running = True
    info_label.config(text="Show a banking gesture to begin")

def next_step():
    global current_step, gesture_locked, current_gesture

    if current_gesture is None:
        gesture_locked = False
        return

    current_step += 1

    if current_step < len(gesture_actions[current_gesture]):
        instruction = gesture_actions[current_gesture][current_step]
        info_label.config(
            text=f"Gesture: {current_gesture}\nStep {current_step+1}\n\n{instruction}"
        )
        speak(instruction)
    else:
        info_label.config(text="Process completed.\nShow next gesture.")
        gesture_locked = False
        current_gesture = None
        current_step = 0

def about_app():
    info_label.config(
        text="SIGNBANK\n\nGesture-based banking assistant\n"
             "Designed for hearing & speech impaired users\n\n"
             "Offline | Secure | Accessible"
    )

def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

# ================= BUTTONS ====================
tk.Button(btn_frame, text="START", width=10, font=("Arial", 14),
          bg="#1E3A8A", fg="white", command=start_detection).pack(side="left", padx=6)

tk.Button(btn_frame, text="NEXT", width=10, font=("Arial", 14),
          bg="#2563EB", fg="white", command=next_step).pack(side="left", padx=6)

tk.Button(btn_frame, text="TEST VOICE", width=10, font=("Arial", 14),
          command=lambda: speak("Voice is working now")).pack(side="left", padx=6)

tk.Button(btn_frame, text="ABOUT", width=10, font=("Arial", 14),
          bg="#059669", fg="white", command=about_app).pack(side="left", padx=6)

tk.Button(btn_frame, text="EXIT", width=10, font=("Arial", 14),
          bg="#DC2626", fg="white", command=exit_app).pack(side="left", padx=6)

# ================= START APP ==================
update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
 