import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

# ================= CONFIG =====================
GESTURES = [
    "OPEN",
    "CLOSE",
    "BALANCE",
    "WITHDRAW",
    "DEPOSIT",
    "UPDATE",
    "HELP",
    "WAIT",
    "DONE"
]

SAMPLES_PER_GESTURE = 300
MODEL_PATH = "model/sign_model.pkl"

# ================= REMOVE OLD MODEL =================
if os.path.exists(MODEL_PATH):
    os.remove(MODEL_PATH)
    print("🗑️ Old model deleted")

# ================= MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# ================= CAMERA =====================
cap = cv2.VideoCapture(0)

data = []

print("\n===== SIGNBANK MODEL TRAINING =====\n")

# ================= DATA COLLECTION =================
for gesture in GESTURES:
    print(f"\n➡️ Prepare to show gesture: {gesture}")
    input("Press ENTER to start collecting samples...")

    count = 0
    while count < SAMPLES_PER_GESTURE:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                row = []
                for lm in hand.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                row.append(gesture)
                data.append(row)
                count += 1

        cv2.putText(
            frame,
            f"{gesture} : {count}/{SAMPLES_PER_GESTURE}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Gesture Training - Press ESC to skip", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    print(f"✅ Collected {count} samples for {gesture}")

# ================= CLEANUP ====================
cap.release()
cv2.destroyAllWindows()

# ================= TRAIN MODEL =================
print("\n⚙️ Training model...")

df = pd.DataFrame(data)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("\n✅ Model trained successfully!")
print(f"📁 Saved at: {MODEL_PATH}")
