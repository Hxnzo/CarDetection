import cv2
import time
import sys

# Constants: 
# Haar cascade file for car detection
CASCADE_FILE = 'cars.xml'  
# Window name for displaying the video
WINDOW_NAME = 'Car Detection'  

def load_cascade(cascade_file):
    """Load the Haar cascade file for car detection."""
    car_cascade = cv2.CascadeClassifier(cascade_file)
    if car_cascade.empty():
        raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
    return car_cascade

def detect_cars(frame, car_cascade):
    """Detect cars in a video frame."""
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    return car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

def draw_rectangles(frame, cars):
    """Draw rectangles around detected cars."""
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

def process_video(video_path, car_cascade):
    """Process the video and display car detection results."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file {video_path}")

    # Get basic video information
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Info: {total_frames} frames, {fps} FPS")

    start_time = time.time()
    frame_counter = 0

    while True:
        # Read the next video frame
        ret, frame = cap.read()  
        if not ret:
            break

        frame_counter += 1
        frame_start_time = time.time()

        # Detect and annotate cars in the frame
        cars = detect_cars(frame, car_cascade)
        draw_rectangles(frame, cars)

        # Display frame details and the detection results
        cv2.putText(frame, f'Frame: {frame_counter}/{total_frames}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Detected Cars: {len(cars)}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW_NAME, frame)

        # Quit the video display on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Log frame processing time
        frame_end_time = time.time()
        frame_processing_time = frame_end_time - frame_start_time
        print(f"Frame {frame_counter}: {frame_processing_time:.2f} seconds")

    # Log total processing time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Elapsed Time: {elapsed_time:.2f} seconds")

    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to initialize the car detection process."""
    if len(sys.argv) < 2:
        print("Usage: python car_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    car_cascade = load_cascade(CASCADE_FILE)
    process_video(video_path, car_cascade)

if __name__ == "__main__":
    main()
