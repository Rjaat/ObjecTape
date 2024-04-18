import torch
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime, timedelta

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

video_path = 'IMG_4185.MOV'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
output_video_path = 'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))  # Set the output frame rate to match input

frame_number = 0
timestamps = {}  # Dictionary to store timestamps for each detected object

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection using YOLOv5
    results = model(frame)

    # Extract bounding boxes and labels
    bboxes = results.xyxy[0].cpu().numpy()

    # Check if any objects are detected
    if len(bboxes) > 0:
        for bbox in bboxes:
            # Extract class index and convert to int
            class_index = int(bbox[-1])

            # Get class name using the index
            class_name = model.names[class_index]

            # Calculate the duration in the original video
            duration_seconds = frame_number / fps
            duration_timestamp = str(timedelta(seconds=duration_seconds))

            # Store the duration in the dictionary
            timestamps[class_name] = duration_timestamp

            # Overlay duration on the frame
            cv2.putText(frame, f'object detected at {duration_timestamp}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

    frame_number += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Print durations for each detected object
for class_name, duration in timestamps.items():
    print(f'{class_name} detected at {duration}')