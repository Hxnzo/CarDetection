# import cv2
# import time
# import sys

# # Constants
# CASCADE_FILE = 'cars.xml'

# def load_cascade(cascade_file):
#     car_cascade = cv2.CascadeClassifier(cascade_file)
#     if car_cascade.empty():
#         raise IOError(f"Unable to load the cascade classifier from {cascade_file}")
#     print("Cascade classifier loaded successfully.")
#     return car_cascade

# def detect_cars(frame, car_cascade):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
#     return cars

# def draw_rectangles(frame, cars):
#     for (x, y, w, h) in cars:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# def process_video(video_path, car_cascade):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise IOError(f"Error opening video file {video_path}")

#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Video Info: {frame_count} frames, {fps:.2f} FPS")

#     start_time = time.time()
#     processed_frames = 0
#     total_detections = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         cars = detect_cars(frame, car_cascade)
#         draw_rectangles(frame, cars)
#         total_detections += len(cars)
#         processed_frames += 1

#         if processed_frames % 10 == 0:  
#             print(f"Processed {processed_frames}/{frame_count} frames...")

#     end_time = time.time()
#     print(f"Processing Time: {end_time - start_time:.2f} seconds")
#     print("Video processing completed.")
#     print(f"Total cars detected in video: {total_detections}")  

#     cap.release()

# def main():
#     if len(sys.argv) < 2:
#         print("Usage: python carDetectionARM64.py <video_path>")
#         sys.exit(1)

#     video_path = sys.argv[1]
#     car_cascade = load_cascade(CASCADE_FILE)
#     process_video(video_path, car_cascade)

# if __name__ == "__main__":
#     main()


import cv2
import time
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python carDetectionARM64.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    
    # Capture frames from a video
    cap = cv2.VideoCapture(video_path)
    
    # Load the car classifier
    car_cascade = cv2.CascadeClassifier('cars.xml')

    print("Car classifier loaded successfully.")
    car_count = 0  # Counter for cars
    num_frames = 0
    i = 0  # Frame count for inference

    # Process the first 500 frames for inference timing
    while i < 500:
        ret, frames = cap.read()
        if not ret:
            print("End of video or error in reading frame.")
            break

        num_frames += 1
        gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

        # Start inference timer
        start = time.time()

        # Detect cars in the frame
        cars = car_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

        # End inference timer
        end = time.time()
        
        print("Inference time: ", end - start)
        
        # Draw rectangles around detected cars
        for (x, y, w, h) in cars:
            cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
            print("Boundary box: car: ", x, y, w, h)
        
        car_count += len(cars)

        i += 1

        # Display frames in a window
        cv2.imshow('Car Detection', frames)
    
        # Wait for Esc key to stop
        if cv2.waitKey(33) == 27:
            break

    # Release resources and close display window
    cap.release()
    cv2.destroyAllWindows()

    print("Total frames processed: ", num_frames)
    print("Total cars detected: ", car_count)

if __name__ == "__main__":
    main()
