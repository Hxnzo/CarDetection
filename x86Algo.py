import cv2
import sys
import os
import time

# Constants
CASCADE_FILE = 'cars.xml'  # Path to the Haar cascade file for vehicle detection
DISPLAY_WINDOW = 'Vehicle Detection'  # Name of the display window

def initialize_cascade(file_path):
    """
    Load and validate the Haar Cascade file.
    :param file_path: Path to the Haar cascade XML file.
    :return: Loaded Haar cascade object.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: {file_path} not found.")
    
    cascade = cv2.CascadeClassifier(file_path)
    if cascade.empty():
        raise IOError(f"Error loading cascade file: {file_path}")
    return cascade

def detect_and_highlight(frame, cascade):
    """
    Detect vehicles in a video frame and draw bounding boxes around them.
    :param frame: The current frame from the video.
    :param cascade: The loaded Haar cascade for detection.
    :return: Number of vehicles detected in the frame.
    """
    # Convert the frame to grayscale for better detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Perform vehicle detection
    detections = cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.2,  # Adjusts how much the image is scaled at each step
        minNeighbors=3    # Minimum neighbors for a rectangle to be considered a detection
    )
    
    # Draw bounding boxes around detected vehicles
    for x, y, w, h in detections:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return len(detections)

def display_video(video_file, cascade):
    """
    Process and display a video with vehicle detection annotations.
    :param video_file: Path to the video file.
    :param cascade: Loaded Haar cascade for detection.
    """
    # Check if the video file exists
    if not os.path.isfile(video_file):
        raise FileNotFoundError(f"Error: Video file '{video_file}' not found.")

    # Open the video file
    video = cv2.VideoCapture(video_file)
    if not video.isOpened():
        raise IOError(f"Error opening video: {video_file}")

    # Retrieve video metadata
    frame_total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    video_fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second
    print(f"Video Info: Total frames: {frame_total}, FPS: {video_fps:.2f}")

    start_time = time.time()  # Start the timer for processing
    processed_frames = 0  # Counter for processed frames

    # Loop through the video frames
    while True:
        success, frame = video.read()  # Read the next frame
        if not success:
            break  # Exit loop if no more frames are available

        processed_frames += 1  # Increment the frame counter

        # Detect vehicles and annotate the frame
        vehicles_detected = detect_and_highlight(frame, cascade)

        # Overlay detection and frame information on the video
        overlay_info = [
            f'Frame: {processed_frames}/{frame_total}',
            f'Vehicles Detected: {vehicles_detected}'
        ]
        for i, text in enumerate(overlay_info):
            cv2.putText(
                frame, text, (10, 30 + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        # Display the annotated frame
        cv2.imshow(DISPLAY_WINDOW, frame)

        # Exit video playback when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting video...")
            break

    # Calculate total processing time
    total_time = time.time() - start_time
    print(f"Processing complete. Total time: {total_time:.2f} seconds.")

    # Release resources and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()

def run_program():
    """
    Main function to handle input and run the program.
    """
    # Ensure the user provides a video file path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python car_detection.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]  # Get the video path from command-line arguments

    try:
        # Initialize the cascade and process the video
        cascade = initialize_cascade(CASCADE_FILE)
        display_video(video_path, cascade)
    except Exception as error:
        # Print any errors that occur during execution
        print(f"Error: {error}")
        sys.exit(1)

# Entry point of the program
if __name__ == "__main__":
    run_program()
