import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import threading
from queue import Queue

# Initialize text-to-speech engine
engine = pyttsx3.init()
speak_queue = Queue()

# Define a thread function for text-to-speech
def speak_text():
    last_spoken = ""
    while True:
        text = speak_queue.get()
        if text is None:  # Stop the thread when None is received
            break
        if text != last_spoken:  # Speak only if the word has changed
            engine.say(text)
            engine.runAndWait()
            last_spoken = text
        speak_queue.task_done()

# Start the speech thread
speech_thread = threading.Thread(target=speak_text, daemon=True)
speech_thread.start()

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Start video capture
cap = cv2.VideoCapture(0)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_faces = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
face_detection = mp_faces.FaceDetection(min_detection_confidence=0.5)

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'K', 
               10: 'J', 11: 'L', 12: 'L', 14: 'M', 16: '0', 17: '1', 19: 'N', 20: 'O', 21: 'P', 
               22: 'P', 23: 'Q', 24: 'R', 25: 'S', 26: '9', 27: '8', 28: '8', 29: '8', 30: '7', 
               31: '6', 32: '5', 33: '4', 34: '3', 35: '2', 36: '1', 37: '0', 38: 'Y', 39: 'X', 
               40: 'W', 41: 'V', 42: 'U', 43: 'T', 44: 'Z'}
while True:
    data_aux = []
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    hand_results = hands.process(frame_rgb)

    # Process faces
    face_results = face_detection.process(frame_rgb)

    # Hand detection and gesture recognition
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect landmarks
            x_ = [landmark.x for landmark in hand_landmarks.landmark]
            y_ = [landmark.y for landmark in hand_landmarks.landmark]
            min_x, min_y = min(x_), min(y_)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min_x)
                data_aux.append(landmark.y - min_y)

        if len(data_aux) < 84:
            data_aux.extend([0] * (84 - len(data_aux)))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        confidence = model.predict_proba([np.asarray(data_aux)])[0].max() * 100

        x1, y1 = int(min(x_) * W) - 10, int(min(y_) * H) - 10
        x2, y2 = int(max(x_) * W) - 10, int(max(y_) * H) - 10
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, f"{predicted_character} ({confidence:.2f}%)", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Speak if confidence is above 70% and word hasn't been spoken already
        if confidence > 60:
            speak_queue.put(predicted_character)

    # Face detection - Fix false hand as face issue
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x1 = int(bboxC.xmin * W)
            y1 = int(bboxC.ymin * H)
            x2 = int((bboxC.xmin + bboxC.width) * W)
            y2 = int((bboxC.ymin + bboxC.height) * H)

            # Adjust criteria to avoid misinterpreting hand as face
            if abs(x2 - x1) > 50 and abs(y2 - y1) > 50:  # A simple threshold for face size
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                score = int(detection.score[0] * 100)
                cv2.putText(frame, f'Face: {score}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand and Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Stop the speech thread
speak_queue.put(None)
speech_thread.join()
