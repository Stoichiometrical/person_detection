import cv2
import os
import time

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the camera (0 represents the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to exit.")

person_detected = False  # Flag to track if a person is currently detected
picture_taken = False    # Flag to ensure the picture is taken only once

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (required for Haar Cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any faces are detected
    if len(faces) > 0:
        if not person_detected:  # If a person was not previously detected, start printing
            person_detected = True
            print("Person detected!")

            # Take a picture and save it only once
            if not picture_taken:
                timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp for the filename
                filename = f"person_detected_{timestamp}.jpg"
                cv2.imwrite(filename, frame)  # Save the current frame as an image
                print(f"Picture saved as {filename}")
                picture_taken = True  # Set the flag to prevent taking another picture
    else:
        if person_detected:  # If a person was previously detected but is now gone, reset flags
            person_detected = False
            picture_taken = False  # Reset the picture flag for the next detection

    # Draw rectangles around detected faces (optional, for visualization)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()