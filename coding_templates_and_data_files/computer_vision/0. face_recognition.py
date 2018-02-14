# Face Recognition

# Importing the Libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Defining a function that will do the detections
def detect(grey, frame):
    """
    The two arguments that are inside of the detect function consist of:
        - grey - the black and white image
        - frame - the image we want to draw the rectangles on
    
    Faces is a tuple of 4 elements: 
        - x, y - these are the coordinates of the upper left corner
        - w, h - width and height of the rectangle
    
    Arguments are as follows: 
        - grey = black and white image 
        - 1.3 = scale factor, how much of the size of the image is going to be reduced
        - 5 = minimum number of neighbour zones that must be accepted
    """
    faces = face_cascade.detectMultiScale(grey, 1.3, 5)
    for (x, y, w, h) in faces:
        # Draw the rectangle
        """
        cv2.rectangle() consists of:
            - The image we want to draw the rectangles on
            - The coordinates of the upper left corner of the rectangle
            - The coordinates of the lower right corner of the rectangle
            - Colour of the rectangle
            - The thickness of the edges of the rectangle
        """
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        ## Eyes
        # Zone of interest which is inside the detector rectangle
        # You need the black and white & coloured versions
        roi_grey = grey[y:y+h, x:x+w]
        roi_colour = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_grey, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_colour, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# Doing some Face Recognition with the webcam
# 0 = Laptop/built in webcam
# 1 = External/plugged in webcam
video_capture = cv2.VideoCapture(0)
while True:
    # Outputs the last frame of the webcam
    _, frame = video_capture.read()
    # Outputs the right shades of grey
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(grey, frame)
    # Shows the animated version of the webcam frames
    cv2.imshow('Video', canvas)
    # Stop the webcam by pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Turns off the webcam
video_capture.release()
# Removes the webcam window
cv2.destroyAllWindows()