import cv2
import os
import time

# Initialize the HOG descriptor for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # Resize the frame for faster processing (optional)
    frame = cv2.resize(frame, (640, 480))

    # Detect people in the frame using HOG
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # Check if any people are detected
    if len(rects) > 0:
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

    # Draw rectangles around detected people (optional, for visualization)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()