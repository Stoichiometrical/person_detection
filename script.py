import cv2

# Load the pre-trained Haar Cascade classifier for full-body detection
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initialize the camera (0 represents the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (required for Haar Cascades)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect bodies in the frame
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If a body is detected, print a message
    if len(bodies) > 0:
        print("Person detected!!")

    # Draw rectangles around detected bodies (optional, for visualization)
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Person Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()


# import cv2
#
# # Load the pre-trained Haar Cascade classifier for full-body detection
# body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
#
# # Initialize the camera (0 represents the default webcam)
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()
#
# print("Press 'q' to exit.")
#
# person_detected = False  # Flag to track if a person is currently detected
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     if not ret:
#         print("Error: Could not read frame.")
#         break
#
#     # Convert the frame to grayscale (required for Haar Cascades)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Detect bodies in the frame
#     bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#     # Check if any bodies are detected
#     if len(bodies) > 0:
#         if not person_detected:  # If a person was not previously detected, start printing
#             person_detected = True
#         print("Person detected!!!")  # Print continuously while a person is detected
#     else:
#         if person_detected:  # If a person was previously detected but is now gone, stop printing
#             person_detected = False
#
#     # Draw rectangles around detected bodies (optional, for visualization)
#     for (x, y, w, h) in bodies:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Person Detection', frame)
#
#     # Exit the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the camera and close all windows
# cap.release()
# cv2.destroyAllWindows()