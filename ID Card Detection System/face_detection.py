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

# Capture video from your webcam for real-time face recognition
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a single frame
    ret, frame = video_capture.read()

    # Find all face locations and face encodings in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match is found, use the name of the known face with the smallest distance
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw a box and label for the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
