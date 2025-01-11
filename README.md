# ROI-Based Person Tracking

This project implements a Region of Interest (ROI)-based person tracking system using YOLOv11 for object detection and tracking. It processes a video stream, detects people within a predefined polygon ROI, and tracks their movements. The output video displays bounding boxes, track IDs, and the count of people within the ROI.

## Features

- **YOLOv11 Model**: Utilizes the YOLOv11 model for accurate and efficient object detection and tracking.
- **ROI-Based Tracking**: Tracks people only within a user-defined polygon region of interest.
- **Bounding Boxes and IDs**: Displays bounding boxes and unique track IDs for detected people.
- **Person Counting**: Counts the number of people within the ROI in real-time.
- **Video Processing**: Processes input video files and saves the annotated output video.
