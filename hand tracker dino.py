import cv2
import mediapipe as mp
import numpy as np
import math
import pyautogui
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

threshold = 0.03
prev_distance = None

last_press_time = 0
press_interval = 0.1

middle_finger_detected_time = None

def put_text_with_outline(img, text, pos, font, scale, color, thickness, outline_color, outline_thickness):
    cv2.putText(img, text, pos, font, scale, outline_color, thickness + outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)

def is_finger_up(hand_landmarks, tip, pip, handedness):
    if tip == mp_hands.HandLandmark.THUMB_TIP:
        wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        thumb_tip_x = hand_landmarks.landmark[tip].x
        return thumb_tip_x < wrist_x if handedness == 'Right' else thumb_tip_x > wrist_x
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y

def is_only_middle_finger_up(hand_landmarks, handedness):
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
            mp_hands.HandLandmark.PINKY_PIP]
    status = [is_finger_up(hand_landmarks, tip, pip, handedness) for tip, pip in zip(tips, pips)]
    return status[2] and all(not s for i, s in enumerate(status) if i != 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(frame_rgb)

    jari_tengah_only_terdeteksi = False
    zoom_in_detected = False

    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
        for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            handedness = results_hands.multi_handedness[idx].classification[0].label

            # check fak
            if is_only_middle_finger_up(hand_landmarks, handedness):
                jari_tengah_only_terdeteksi = True
                if middle_finger_detected_time is None:
                    middle_finger_detected_time = time.time()
            else:
                middle_finger_detected_time = None

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = math.sqrt(
                (thumb_tip.x - index_tip.x) ** 2 +
                (thumb_tip.y - index_tip.y) ** 2
            )
            if prev_distance is not None:
                if distance - prev_distance > threshold:
                    zoom_in_detected = True
                    now = time.time()
                    if now - last_press_time > press_interval:
                        pyautogui.press('space')
                        print('lompat cik')
                        status_text += "Lompat!"
                        last_press_time = now
            prev_distance = distance

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )
    else:
        middle_finger_detected_time = None

    status_text = "Status: "
    if zoom_in_detected:
        status_text += "Lompat!"

    put_text_with_outline(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (255, 255, 255), 2, (0, 0, 0), 3)

    cv2.imshow('Hand Tracker Dino Kikok', frame)

    if jari_tengah_only_terdeteksi:
        if middle_finger_detected_time and (time.time() - middle_finger_detected_time) >= 0.5:
            print("pakyu coding.")
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
 