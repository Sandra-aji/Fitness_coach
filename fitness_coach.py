import cv2
import mediapipe as mp
import time
import os
import numpy as np

# 🔹 MODEL PATH
MODEL_PATH = "C:/Users/sandr/fitness/pose_landmarker.task"
print("Model exists:", os.path.exists(MODEL_PATH))

# 🔹 ANGLE FUNCTION
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)
counter = 0
stage = None
# Use average of both legs (more stable)
avg_angle = (left_angle + right_angle) / 2

# DOWN position
if avg_angle < 90:
    stage = "down"

# UP position (rep complete)
if avg_angle > 160 and stage == "down":
    stage = "up"
    counter += 1

# 🔹 MEDIAPIPE SETUP
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = PoseLandmarker.create_from_options(options)

# 🔹 OPEN CAMERA
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        # =========================
        # LEFT LEG
        # =========================
        l_hip = lm[23]
        l_knee = lm[25]
        l_ankle = lm[27]

        l_hip_coords = (l_hip.x * frame.shape[1], l_hip.y * frame.shape[0])
        l_knee_coords = (l_knee.x * frame.shape[1], l_knee.y * frame.shape[0])
        l_ankle_coords = (l_ankle.x * frame.shape[1], l_ankle.y * frame.shape[0])

        left_angle = calculate_angle(l_hip_coords, l_knee_coords, l_ankle_coords)

        # =========================
        # RIGHT LEG
        # =========================
        r_hip = lm[24]
        r_knee = lm[26]
        r_ankle = lm[28]

        r_hip_coords = (r_hip.x * frame.shape[1], r_hip.y * frame.shape[0])
        r_knee_coords = (r_knee.x * frame.shape[1], r_knee.y * frame.shape[0])
        r_ankle_coords = (r_ankle.x * frame.shape[1], r_ankle.y * frame.shape[0])

        right_angle = calculate_angle(r_hip_coords, r_knee_coords, r_ankle_coords)

        # =========================
        # DRAW LANDMARKS
        # =========================
        for l in lm:
            x = int(l.x * frame.shape[1])
            y = int(l.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # =========================
        # DISPLAY ANGLES
        # =========================
        cv2.putText(frame, f"L: {int(left_angle)}",
                    (int(l_knee_coords[0]), int(l_knee_coords[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(frame, f"R: {int(right_angle)}",
                    (int(r_knee_coords[0]), int(r_knee_coords[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(frame, f"Reps: {counter}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5, (0, 0, 255), 3)

    cv2.imshow("AI Fitness Coach - Dual Angle", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()