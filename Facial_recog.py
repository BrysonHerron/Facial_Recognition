import cv2
import time
import winsound

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

start_time = None
detected_face = False

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        detected_face = True
    
    # Display the frame with face detections
    cv2.imshow('Face Detection', frame)
    
    # Check if a face has been detected
    if detected_face:
        # Start the timer if not already started
        if start_time is None:
            start_time = time.time()
        # Check if five seconds have elapsed
        elif time.time() - start_time > 5:
            winsound.Beep(1000, 500)  # Play a beep sound
            detected_face = False
            start_time = None
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()