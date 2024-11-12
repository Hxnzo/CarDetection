import cv2
import time

# Load the video
video_path = "path_to_your_downloaded_video.mp4"
cap = cv2.VideoCapture(video_path)

# Load a pre-trained Haar Cascade for object detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Get total frames for progress calculation
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Start processing and track time
start_time = time.time()
frame_count = 0
interval = 10  # Print progress every 10 frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (required for Haar cascade)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    objects = cascade.detectMultiScale(gray, 1.1, 4)

    # Process each detected object
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    frame_count += 1

    # Print progress every 'interval' frames
    if frame_count % interval == 0:
        elapsed_time = time.time() - start_time
        percent_complete = (frame_count / total_frames) * 100
        estimated_total_time = (elapsed_time / frame_count) * total_frames
        estimated_time_left = estimated_total_time - elapsed_time
        print(f"Processed {frame_count}/{total_frames} frames "
              f"({percent_complete:.2f}% complete) - "
              f"Estimated time left: {estimated_time_left:.2f} seconds")

cap.release()
end_time = time.time()

# Calculate final processing time
processing_time = end_time - start_time
print(f"Processed all {frame_count} frames in {processing_time:.2f} seconds")
