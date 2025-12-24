import cv2
import numpy as np
from keras.models import model_from_json


# Loading trained model
with open("facialemotionmodel.json", "r") as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("facialemotionmodel.h5")


# "Haar Cascade" (Viola) for face detection
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Helper function - preprocess face
def preprocess_face(face_img):
    face_array = np.array(face_img).reshape(1, 48, 48, 1)
    return face_array.astype("float32") / 255.0

# Emotion classes
emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}


# Start video capture(future scopre - won't work for now)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        processed_face = preprocess_face(roi_gray)
        prediction = emotion_model.predict(processed_face, verbose=0)
        emotion_label = emotion_dict[int(np.argmax(prediction))]

        # Draw rectangle and put text (ref Viola haar research)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            frame,
            emotion_label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow("Facial Emotion Recognition", frame)

    # 'q' = exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
