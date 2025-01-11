import cv2
import numpy as np
from ultralytics import YOLO
import cvzone 

# Load the YOLOv11 model
model = YOLO("yolo11s.pt")
names = model.names

# open the video file 
                   
cap = cv2.VideoCapture('sample.mp4')

# Get the frame height and width from the video

frame_width =  int(cap.get(3))
frame_height =  int(cap.get(4))

# Initialize video writer to save the output
output_video_path = "test/output_viedo.mp4"  
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

# Define the polygon area for ROI using your provided coordinates
area = [(524, 282), (152, 331), (243, 459), (834, 427)]

# main loop to process each frame of the video 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    result = model.track(frame, persist=True)

    person_count = 0

    if result[0].boxes is not None and result[0].boxes.id is not None:
        boxes = result[0].boxes.xyxy.int().cpu().tolist()
        class_ids = result[0].boxes.cls.int().cpu().tolist()
        track_ids = result[0].boxes.id.int().cpu().tolist()
        confidences = result[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, confidence in zip(boxes, class_ids, track_ids, confidences):
            c = names[class_id]
            x1, y1, x2, y2 = box

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),(0, 255, 0), 2)

            # Display class name and track ID on the bounding box
            label = f"{track_id}"
            cvzone.putTextRect(frame, label, (x1, y1 -10), scale=1, thickness=2, colorR=(0, 255, 0))
            # Check if the object is person 
            if "person" in c:
                # Calculate the center of the bounding box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # check if the center is inside the ROI
                point_result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if point_result >= 0:  # if the center is inside the ROI
                    person_count += 1
                    cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
    
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)

    # Display the person count
    cvzone.putTextRect(frame, f"person in Roi : {person_count}", (50,50), scale=2, thickness=3, colorR=(0, 0, 255))

    video_writer.write(frame)

    cv2.imshow('Output', frame)

    if cv2.waitKey(1)==27:
        break


cap.release() 
video_writer.release()
cv2.destroyAllWindows()