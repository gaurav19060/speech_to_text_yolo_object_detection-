from ultralytics import YOLO
import cv2
import time
import speech_recognition as sr

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

def run_detection_for_10_seconds():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv8 Detection', 640, 480)  # Set initial window size

    start_time = time.time()
    while time.time() - start_time < 10:  # Run detection for 10 seconds
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Run YOLOv8 on the captured frame
        results = model(frame)

        # Annotate the frame with detected objects
        for det in results[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Bounding box coordinates
            class_id = int(det.cls[0])  # Get the class ID
            label = model.names[class_id]  # Map class ID to the class name
            confidence = det.conf[0].item()  # Confidence score as a float

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for detected objects
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label with confidence

        # Display the frame with detections
        cv2.imshow('YOLOv8 Detection', frame)

        # Exit on pressing 'q' during the 10-second interval
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close windows after 10 seconds or if 'q' is pressed
    cap.release()
    cv2.destroyAllWindows()

def listen_for_command():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command... (say 'What do you see' to start detection)")
        recognizer.adjust_for_ambient_noise(source)  # Adjusts for background noise
        audio = recognizer.listen(source)

    try:
        # Recognize the command using Google Speech Recognition
        command = recognizer.recognize_google(audio).lower()
        print(f"Recognized command: {command}")
        return command
    except sr.UnknownValueError:
        print("Could not understand the audio")
    except sr.RequestError:
        print("Speech Recognition service is not available")
    return ""

def main():
    while True:
        command = listen_for_command()
        
        if "what do you see" in command:
            print("Running detection for 10 seconds...")
            run_detection_for_10_seconds()
        
        elif "quit" in command:
            print("Exiting program.")
            break

        else:
            print("Invalid command. Say 'What do you see' to start or 'quit' to exit.")

if __name__ == "__main__":
    main()
