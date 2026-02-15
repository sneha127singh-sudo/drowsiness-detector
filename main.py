import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import csv
import pyttsx3

# ------------------ INITIALIZE ------------------
pygame.mixer.init()

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def play_alarm():
    pygame.mixer.music.load("alarm.wav")
    pygame.mixer.music.play(-1)

def stop_alarm():
    pygame.mixer.music.stop()

def speak_warning():
    engine.say("Wake up. You are drowsy.")
    engine.runAndWait()

def log_event(event):
    with open("fatigue_log.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%H:%M:%S"), event])

# ------------------ VARIABLES ------------------
alarm_on = False
voice_given = False
fatigue_score = 0
COUNTER = 0
no_face_counter = 0

start_time = time.time()
blink_timestamps = []

# Adaptive calibration
calibration_frames = 0
ear_sum = 0
EYE_AR_THRESH = 0
CALIBRATION_TIME = 100

EYE_AR_CONSEC_FRAMES = 15

# ------------------ MEDIAPIPE ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ------------------ FUNCTIONS ------------------
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]

    vertical1 = np.linalg.norm(np.array([p2.x, p2.y]) - np.array([p6.x, p6.y]))
    vertical2 = np.linalg.norm(np.array([p3.x, p3.y]) - np.array([p5.x, p5.y]))
    horizontal = np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p4.x, p4.y]))

    return (vertical1 + vertical2) / (2.0 * horizontal)

# ------------------ CAMERA ------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        no_face_counter = 0

        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # -------- EYE CALCULATION --------
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0

            # -------- AUTO CALIBRATION --------
            if calibration_frames < CALIBRATION_TIME:
                ear_sum += ear
                calibration_frames += 1
                cv2.putText(frame, "Calibrating... Keep eyes open", (150, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

                if calibration_frames == CALIBRATION_TIME:
                    avg_ear = ear_sum / CALIBRATION_TIME
                    EYE_AR_THRESH = avg_ear * 0.75

            else:
                # -------- EYE DETECTION --------
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    fatigue_score += 1

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        if not alarm_on:
                            play_alarm()
                            speak_warning()
                            alarm_on = True
                            voice_given = True
                            log_event("Microsleep")
                else:
                    if COUNTER > 2:
                        blink_timestamps.append(time.time())
                    COUNTER = 0

                    if alarm_on:
                        stop_alarm()
                        alarm_on = False
                        voice_given = False

                # -------- YAWN DETECTION --------
                top_lip = landmarks[13]
                bottom_lip = landmarks[14]
                mouth_open = abs(top_lip.y - bottom_lip.y)

                if mouth_open > 0.05:
                    fatigue_score += 2
                    cv2.putText(frame, "Yawning!", (30, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    log_event("Yawning")

                # -------- HEAD NOD --------
                nose = landmarks[1]
                if nose.y > 0.6:
                    fatigue_score += 3
                    cv2.putText(frame, "Head Dropping!", (30, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    log_event("Head Nod")

    else:
        no_face_counter += 1
        if no_face_counter > 40:
            cv2.putText(frame, "Face Not Detected!", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            log_event("Face Missing")

    # -------- FATIGUE DECAY --------
    fatigue_score = max(0, fatigue_score - 0.2)

    # -------- ROLLING BLINK RATE (LAST 60 SEC) --------
    current_time = time.time()
    blink_timestamps = [t for t in blink_timestamps if current_time - t <= 60]
    blink_rate = len(blink_timestamps)

    # -------- FATIGUE LEVEL --------
    if fatigue_score > 60:
        level = "HIGH"
        color = (0,0,255)
    elif fatigue_score > 30:
        level = "MEDIUM"
        color = (0,255,255)
    else:
        level = "LOW"
        color = (0,255,0)

    # -------- DASHBOARD --------
    cv2.putText(frame, f"Fatigue Score: {int(fatigue_score)}", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Level: {level}", (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, f"Blink Rate: {blink_rate} /min", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    elapsed_time = int(time.time() - start_time)
    cv2.putText(frame, f"Time: {elapsed_time}s", (30, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Smart Driver Fatigue Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

