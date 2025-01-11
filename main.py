import json
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Load the YOLOv11 model with specified weights
yolo_model = YOLO("yolo11s.pt")  # Replace "yolo11s.pt" with the path to your model weights
class_names = yolo_model.names  # Get class names from the model

# Load the configuration file for polygon area
with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Define the polygon area for Region of Interest (ROI) from the configuration file
roi_polygon = config.get("polygon_area", [])
if not roi_polygon:
    raise ValueError("Polygon area is not defined in the configuration file.")
if len(roi_polygon) < 3:
    raise ValueError("Polygon area must have at least three points.")

# Open the video file for processing
video_path = 'sample.mp4'  # Input video file path
video_capture = cv2.VideoCapture(video_path)

# Get the frame dimensions (width and height) from the video
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writer to save the processed output
output_video_path = "test/output_video.mp4"  # Output video file path
output_fps = video_capture.get(cv2.CAP_PROP_FPS)  # Extract FPS from input video
video_codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter(output_video_path, video_codec, output_fps, (frame_width, frame_height))

# Main loop to process each frame of the video
while True:
    ret, frame = video_capture.read()  # Read the next frame from the video
    if not ret:
        break  # Exit loop if no more frames are available

    # Resize the frame for consistent processing dimensions
    resized_frame = cv2.resize(frame, (1020, 500))

    # Perform object tracking with YOLO
    tracking_results = yolo_model.track(resized_frame, persist=True)

    # Initialize person count for the current frame
    person_count_in_roi = 0

    # Check if any bounding boxes were detected
    if tracking_results[0].boxes is not None and tracking_results[0].boxes.id is not None:
        bounding_boxes = tracking_results[0].boxes.xyxy.int().cpu().tolist()  # Bounding box coordinates
        class_ids = tracking_results[0].boxes.cls.int().cpu().tolist()  # Class IDs for detected objects
        track_ids = tracking_results[0].boxes.id.int().cpu().tolist()  # Track IDs for detected objects
        confidences = tracking_results[0].boxes.conf.cpu().tolist()  # Confidence scores

        # Iterate through detected objects and process their details
        for box, class_id, track_id, confidence in zip(bounding_boxes, class_ids, track_ids, confidences):
            detected_class = class_names[class_id]  # Get the class name
            x1, y1, x2, y2 = box  # Bounding box coordinates


            # Check if the detected object is a person
            if "person" in detected_class:
                # Calculate the center of the bounding box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                # Check if the center of the bounding box is inside the ROI polygon
                is_inside_roi = cv2.pointPolygonTest(np.array(roi_polygon, np.int32), (center_x, center_y), False)
                if is_inside_roi >= 0:  # Object center is inside the ROI
                    person_count_in_roi += 1

                # Draw a bounding box around the detected object
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # Display the track ID on the bounding box
                    label_text = f"ID: {track_id}"
                    cvzone.putTextRect(resized_frame, label_text, (x1, y1 - 10), scale=1, thickness=2, colorR=(0, 255, 0))
                    # Mark the center point of the bounding box 
                    cv2.circle(resized_frame, (center_x, center_y), 4, (255, 0, 0), -1)  # Mark the center point

    # Draw the ROI polygon on the frame
    cv2.polylines(resized_frame, [np.array(roi_polygon, np.int32)], isClosed=True, color=(255, 0, 255), thickness=2)

    # Display the person count within the ROI
    cvzone.putTextRect(resized_frame, f"People in ROI: {person_count_in_roi}", (50, 50), scale=2, thickness=3, colorR=(0, 0, 255))

    # Write the processed frame to the output video
    video_writer.write(resized_frame)

    # Display the frame with annotations
    cv2.imshow('Processed Output', resized_frame)

    # Break the loop if the Escape key (ESC) is pressed
    if cv2.waitKey(1) == 27:
        break

# Release resources and clean up
video_capture.release()
video_writer.release()
cv2.destroyAllWindows()
