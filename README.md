# Smart Adaptive Traffic Light System

Final Project - Computer Vision (COMP7116001)
BINUS University

## Overview

This project implements a simple adaptive traffic light system using classical Computer Vision techniques. The application detects moving vehicles from a traffic video, counts the number of vehicles in each lane, and adjusts the green light duration based on the traffic density.

The project does not use Deep Learning, CNN, YOLO, or any pre-trained model. All vehicle detection is performed using image processing methods available in OpenCV.

## Features

* Detects moving vehicles from a traffic video.
* Counts vehicles in predefined traffic lanes.
* Adjusts green light duration according to vehicle density.
* Displays the simulation through a Streamlit interface.
* Allows users to modify detection parameters for different videos.

## Technologies

* Python
* OpenCV
* Streamlit
* NumPy

## Project Structure

```
smart-traffic-light/
│── app.py
│── detector.py
│── traffic_controller.py
│── config.py
│── requirements.txt
└── data/
```

## Installation

Install the required libraries by running:

```bash
pip install -r requirements.txt
```

After that, start the application with:

```bash
streamlit run app.py
```

The application will open automatically in your browser at:

```
http://localhost:8501
```

## How to Use

1. Launch the application.
2. Upload a traffic image or video 
3. Adjust the detection parameters if necessary.
4. Press the **Start** button to begin the simulation.
5. The system will detect vehicles, count them for each lane, and update the traffic light duration automatically.

## Processing Flow

```
Input Video
      ↓
Background Subtraction
      ↓
Thresholding
      ↓
Morphological Operations
      ↓
Contour Detection
      ↓
Vehicle Counting
      ↓
Adaptive Traffic Light Control
```

## Dataset

The system was developed and tested using traffic videos from the UA-DETRAC dataset. Other traffic videos captured by a fixed camera can also be used, provided that the camera remains stationary during recording.

## Notes

This project was developed as the final project for the Computer Vision course at BINUS University. The implementation focuses on classical image processing techniques, making it suitable for academic purposes where the use of Deep Learning models is not allowed.
