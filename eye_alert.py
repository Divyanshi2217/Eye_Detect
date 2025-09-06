import cv2
import mediapipe as mp
import math
import threading
import pygame

# --- Functions ---
def eye_aspect_ratio(landmarks, eye_indices):
    A = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    B = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    C = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
    ear = (A + B) / (2.0 * C)
    return ear

def play_alert_sound():
    global alert_playing
    pygame.mixer.music.play(-1)  # loop continuously
    alert_playing = True

def stop_alert_sound():
    global alert_playing
    pygame.mixer.music.stop()
    alert_playing = False

# --- Initialize Pygame for sound ---
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # use a WAV file

# --- Mediapipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,            # optional, gives more detailed eye landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 10

frame_counter = 0
alert_playing = False

# --- Webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    
    eyes_closed = False
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = []
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks.append((x, y))
            
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Check if eyes are closed
            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSEC_FRAMES:
                    eyes_closed = True
                    cv2.putText(frame, "ALERT! Eyes Closed!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # Red rectangle background
                    cv2.rectangle(frame, (0,0), (w,h), (0,0,255), thickness=5)
                    # Start alert sound if not playing
                    if not alert_playing:
                        threading.Thread(target=play_alert_sound, daemon=True).start()
            else:
                frame_counter = 0
                # Eyes open: green rectangle and stop alert
                cv2.rectangle(frame, (0,0), (w,h), (0,255,0), thickness=5)
                if alert_playing:
                    stop_alert_sound()
    
    cv2.imshow("Eye Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        if alert_playing:
            stop_alert_sound()
        break

cap.release()
cv2.destroyAllWindows()
