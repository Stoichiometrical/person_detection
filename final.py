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

person_detected = False       # Flag to track if a person is currently detected
picture_taken = False         # Flag to ensure the picture is taken only once
recording = False             # Flag to track if recording is in progress
out = None                    # Video writer object
start_time = None             # Start time of recording

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
        if not person_detected:  # If a person was not previously detected
            person_detected = True
            print("Person detected!")

            # Take a picture and save it only once
            if not picture_taken:
                timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp for the filename
                picture_filename = f"person_detected_{timestamp}.jpg"
                cv2.imwrite(picture_filename, frame)  # Save the current frame as an image
                print(f"Picture saved as {picture_filename}")
                picture_taken = True

            # Start recording if not already recording
            if not recording:
                recording = True
                video_filename = f"person_video_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define codec
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                start_time = time.time()  # Record the start time
                print(f"Started recording video: {video_filename}")

        # Write the frame to the video file if recording is active
        if recording and out is not None:
            out.write(frame)

    else:
        if person_detected:  # If a person was previously detected but is now gone
            # Ensure at least 10 seconds have passed before stopping the recording
            if start_time is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time >= 10:  # Stop recording after 10 seconds
                    if recording and out is not None:
                        out.release()  # Stop recording
                        out = None
                        recording = False
                        start_time = None
                        print("Stopped recording video after minimum duration.")

            # Reset flags for the next detection cycle
            person_detected = False
            picture_taken = False

    # Continue recording even if the person leaves the frame before 10 seconds
    if recording and out is not None:
        if start_time is not None:
            elapsed_time = time.time() - start_time
            if elapsed_time >= 10 and not person_detected:  # Stop recording if no one is in the frame after 10 seconds
                out.release()  # Stop recording
                out = None
                recording = False
                start_time = None
                print("Stopped recording video after minimum duration with no one in frame.")

    # Draw rectangles around detected people (optional, for visualization)
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, stop recording if necessary, and close all windows
if out is not None:
    out.release()
cap.release()
cv2.destroyAllWindows()