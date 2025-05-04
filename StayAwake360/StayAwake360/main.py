import cv2
import numpy as np
import dlib
import os
import threading
from imutils import face_utils
from playsound import playsound

# ==== CONFIGURATION ====
EAR_THRESHOLD_SLEEP = 0.21 #  Threshold for EAR to determine if the user is sleeping
EAR_THRESHOLD_DROWSY = 0.25 #  Threshold for EAR to determine if the user is drowsy
SLEEP_FRAME_LIMIT = 6 #  Number of frames to determine if the user is sleeping
DROWSY_FRAME_LIMIT = 6 #  Number of frames to determine if the user is drowsy
ACTIVE_FRAME_LIMIT = 6 #  Number of frames to determine if the user is active
MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
ALERT_SOUND = "alert.mp3" #  File name for the alert sound

# ==== GLOBALS ====
alert_playing = False #  Flag to check if the alert sound is playing

# ==== FUNCTIONS ====
def sound_alert():
    global alert_playing
    alert_playing = True
    try:
        playsound(ALERT_SOUND)
    except Exception as e:
        print(f"Failed to play sound: {e}")
    alert_playing = False

# Define a function to calculate the eye aspect ratio
# The eye aspect ratio is a measurement of the distance between the inner and outer corners of the eye
# It is used to detect if a person is blinking
def eye_aspect_ratio(eye):
    # Calculate the distance between the inner and outer corners of the left eye
    A = np.linalg.norm(eye[1] - eye[5])
    # Calculate the distance between the inner and outer corners of the right eye
    B = np.linalg.norm(eye[2] - eye[4])
    # Calculate the distance between the inner and outer corners of the eyes
    C = np.linalg.norm(eye[0] - eye[3])
    # Calculate the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # Return the eye aspect ratio
    return ear

def load_dlib_models():
    # Get the directory of the current file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Join the base directory with the model filename to get the full path to the model file
    model_path = os.path.join(base_dir, MODEL_FILENAME)

    # Check if the model file exists
    if not os.path.exists(model_path):
        # If the model file does not exist, raise a FileNotFoundError
        raise FileNotFoundError(f"Required model file not found at: {model_path}")

    # Load the dlib face detector
    detector = dlib.get_frontal_face_detector()
    # Load the dlib shape predictor
    predictor = dlib.shape_predictor(model_path)
    # Return the detector and predictor
    return detector, predictor

# ==== MAIN PROGRAM ====
def main():
    # Declare a global variable to keep track of whether an alert is currently playing
    global alert_playing

    # Load the dlib models for face detection and facial landmark prediction
    detector, predictor = load_dlib_models()
    # Open the webcam
    cap = cv2.VideoCapture(0)

    # Initialize variables to keep track of the state of the user
    sleep = drowsy = active = 0
    status = ""
    color = (0, 0, 52)

    # Loop to continuously capture frames from the webcam
    while True:
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        face_frame = frame.copy()

        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # Check if the user is sleeping
            if ear < EAR_THRESHOLD_SLEEP:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > SLEEP_FRAME_LIMIT:
                    status = "SLEEPING (~_~)!"
                    color = (255, 0, 0)
                    if not alert_playing:
                        threading.Thread(target=sound_alert).start()
            # Check if the user is drowsy
            elif ear < EAR_THRESHOLD_DROWSY:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > DROWSY_FRAME_LIMIT:
                    status = "Drowsy (T_T)!"
                    color = (0, 0, 255)
                    if not alert_playing:
                        threading.Thread(target=sound_alert).start()
            # Check if the user is active
            else:
                sleep = 0
                drowsy = 0
                active += 1
                if active > ACTIVE_FRAME_LIMIT:
                    status = "Active (^-^)"
                    color = (0, 255, 0)

            # Draw status
            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
            cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

            # Draw landmarks
            for (x, y) in shape:
                cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        cv2.imshow("StayAwake360", frame)

        if len(faces) > 0:
            cv2.imshow("Facial Landmarks", face_frame)

        if cv2.waitKey(1) == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()

# ==== ENTRY POINT ====
if __name__ == "__main__":
    main()
