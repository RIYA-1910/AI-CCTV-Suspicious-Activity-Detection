## AI-Based CCTV Suspicious Activity Detection and Real-Time Alert System

A real-time video surveillance system built with Python that uses YOLOv8 object detection 
and ByteTrack multi-object tracking to automatically monitor restricted areas and raise alerts.

### What it does
- Detects when a person enters a restricted zone
- Raises a loitering alert if a person stays for more than 20 seconds
- Plays a siren alarm if the person stays for more than 25 seconds
- Detects unattended objects (bags, bottles, laptops) left in the zone for 60+ seconds
- Saves evidence screenshots with timestamps for every alert
- Logs all incidents in a SQLite database

### Tech Stack
Python | YOLOv8 | OpenCV | ByteTrack | Streamlit | SQLite

### How to run
1. Install dependencies: pip install ultralytics opencv-python streamlit pandas
2. Run the app: streamlit run app.py
3. Upload a video or connect a webcam, draw the restricted zone, and start detection
