import streamlit as st
import torch
import cv2
import numpy as np
from datetime import timedelta
import os
from io import BytesIO

# Load YOLOv5 model
@st.cache(allow_output_mutation=True)
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)

# Function to perform object detection and cropping
def objectape(video_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_video_path = os.path.join("objectape_output.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    timestamps = []

    with st.spinner("Detecting objects and cropping the video..."):
        progress_bar = st.progress(0)

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
                    # Calculate the duration in the original video
                    duration_seconds = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                    duration_timestamp = str(timedelta(seconds=duration_seconds))

                    # Draw bounding box and timestamp on the frame
                    label = f"Object detected at {duration_timestamp}"
                    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Write the frame to the output video
                    out.write(frame)
                    
                    timestamps.append((duration_timestamp, frame.copy()))

            # Update progress bar
            progress = int((cap.get(cv2.CAP_PROP_POS_FRAMES) / total_frames) * 100)
            progress_bar.progress(progress)

    cap.release()
    out.release()

    return timestamps, output_video_path

# Streamlit UI
st.set_page_config(page_title="ObjecTape: Object Detection and Video Cropping Tool", layout="centered")

st.markdown(
"""
<style>
    .reportview-container {
        background: linear-gradient(to right, #ffffff, #b3ecff);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #b3ecff, #ffffff);
    }
</style>
""",
unsafe_allow_html=True
)

st.title("ObjecTape")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "MOV"])

if uploaded_file is not None:
    video_bytes = uploaded_file.read()
    st.video(BytesIO(video_bytes))

    if st.button("Detect Objects and Crop"):
        st.markdown("---")
        st.info("Please wait while ObjecTape detects objects and crops the video...")

        # Save uploaded file to disk
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_bytes)

        timestamps, output_video_path = objectape("uploaded_video.mp4")

        st.success("Object detection and cropping completed!")

        # Display recombined video with timestamps
        st.subheader("Detected Objects Video")
        st.video(output_video_path)
