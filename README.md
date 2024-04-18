# ObjecTape: Object Detection and Video Cropping Tool

ObjecTape is a Streamlit web application that utilizes YOLOv5 model for object detection in videos and provides the functionality to crop the detected objects into a new video with timestamps.

## Features

- Upload a video file (supported formats: mp4, MOV).
- Detect objects in the video using YOLOv5 model.
- Crop the detected objects into a new video.
- Display the new video with timestamps indicating when each object was detected.

## Usage

1. Clone the repository:

      ```git clone https://github.com/rjaat/ObjecTape.git```
   
      ```cd ObjecTape```


2. Install the required dependencies:

    ```pip install -r requirements.txt```

3. Run the Streamlit app:

    ```streamlit run script.py```

  Upload a video file using the file uploader.
    Click on the "Detect Objects and Crop" button to start the object detection process.
    Once the process is complete, the new video with timestamps will be displayed.

Requirements

    Python 3.6+
    Streamlit
    PyTorch
    OpenCV
    NumPy

Credits

    YOLOv5: Ultralytics YOLOv5
    Streamlit: Streamlit
