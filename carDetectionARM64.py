import cv2
import time
import sys

# Constant: File containing the pre-trained Haar cascade for car detection
CASCADE_FILE = 'cars.xml'  

# Load the Haar cascade classifier
def load_cascade(cascade_file):
    car_cascade = cv2.CascadeClassifier(cascade_file)
    if car_cascade.empty():
        raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
    print("Cascade classifier loaded successfully.")
    return car_cascade

# Detect cars in a given video frame
def detect_cars(frame, car_cascade):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return cars

# Draw bounding boxes around detected cars
def draw_rectangles(frame, cars):
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Process the video file frame by frame
def process_video(video_path, car_cascade):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    # Display basic video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {frame_count} frames, {fps:.2f} FPS")

    start_time = time.time()
    processed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect and annotate cars in the frame
        cars = detect_cars(frame, car_cascade)
        draw_rectangles(frame, cars)
        processed_frames += 1

        # Log progress every 10 frames
        if processed_frames % 10 == 0:
            print(f"Processed {processed_frames}/{frame_count} frames...")

    # Display processing time after completing
    end_time = time.time()
    print(f"Processing Time: {end_time - start_time:.2f} seconds")
    print("Video processing completed.")

    cap.release()

# Main function to initialize and run the detection process
def main():
    if len(sys.argv) < 2:
        print("Usage: python object_detection_cli.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    car_cascade = load_cascade(CASCADE_FILE)
    process_video(video_path, car_cascade)

if __name__ == "__main__":
    main()
