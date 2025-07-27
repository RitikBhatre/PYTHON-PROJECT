import face_recognition
import cv2
import os

# Directory where you store face images for registration
registration_dir = "registration_images"

# Create an array to hold the known face encodings and names
known_face_encodings = []
known_face_names = []

# Load images for registration
for filename in os.listdir(registration_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(registration_dir, filename)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)[0]

        # Extract the name from the filename (assuming the filename is in the format "Name.jpg")
        name = os.path.splitext(filename)[0]

        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

# Capture video from your webcam for real-time face registration
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Find all face locations in the frame
    face_locations = face_recognition.face_locations(frame)

    for (top, right, bottom, left) in face_locations:
        # Prompt the user to input their name for registration
        name = input("Enter your name for registration: ")

        if name:
            # Save the face image with the provided name
            face_image = frame[top:bottom, left:right]
            image_filename = os.path.join(registration_dir, f"{name}.jpg")
            cv2.imwrite(image_filename, face_image)
            print(f"Face registered as {name}")

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
