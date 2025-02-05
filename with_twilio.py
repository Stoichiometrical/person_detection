import cv2
import os
import time


def send_whatsapp_alert(message_body, to_number):
    """Send a WhatsApp message (text only, no images)."""
    try:
       
        print(f"Person detected}")
    except Exception as e:
        print(f"Error sending WhatsApp message: {e}")

# Initialize the HOG descriptor for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize the camera (0 represents the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to exit.")

person_detected = False  # Flag to track person detection
alert_sent = False       # Flag to prevent repeated WhatsApp messages

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Resize for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Detect people in the frame
    (rects, _) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    if len(rects) > 0:  # If person is detected
        if not person_detected:
            person_detected = True
            print("Person detected!")

            # Save the image
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"person_detected_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Picture saved as {filename}")

            # Send WhatsApp alert only once per detection
            if not alert_sent:
                send_whatsapp_alert("Person detected in the room!")
                alert_sent = True  # Prevent duplicate alerts

    else:
        if person_detected:  # Reset when person leaves
            person_detected = False
            alert_sent = False  # Allow new alerts

    # Draw detection rectangles
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Person Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
