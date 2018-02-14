# Smile Detector

# Importing the Libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Defining a function that will do the detections
def detect(grey, frame):
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_grey = grey[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 22)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        # Smile detection
        smiles = smile_cascade.detectMultiScale(roi_grey, 1.7, 22)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_colour, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(grey, frame)
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Turns off the webcam
video_capture.release()
# Removes the webcam window
cv2.destroyAllWindows()