# Voice-Activated Object Detection with YOLOv8

This project uses YOLOv8 for real-time object detection controlled by voice commands. When you say "What do you see," the system captures objects detected in a 10-second video feed from the webcam. The project utilizes `speech_recognition` for voice input and `OpenCV` for displaying the live video feed.

## Features

- **Voice Activation**: Start object detection with the command "What do you see."
- **Real-Time Detection**: YOLOv8 detects and displays objects on the video feed in real-time.
- **Automatic Exit**: The detection runs for 10 seconds per command, then automatically returns to the command prompt.

## Requirements

- Python 3.x
- `ultralytics` (for YOLOv8)
- `opencv-python`
- `SpeechRecognition`
- `PyAudio`

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Voice-Activated-Object-Detection
