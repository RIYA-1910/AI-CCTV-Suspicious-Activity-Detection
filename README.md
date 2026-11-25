# AI-Based CCTV Suspicious Activity Detection

A real-time CCTV monitoring dashboard that detects people entering restricted areas,
identifies suspicious loitering, and detects unattended objects using YOLOv8,
ByteTrack, OpenCV, and Streamlit.

Each alert can generate a timestamped evidence image, browser or system sound, and
a record in a local SQLite database.

## Features

- Person detection and multi-object tracking
- Configurable restricted zone drawn as a percentage of the video frame
- Restricted-zone entry alerts
- Loitering detection after a configurable time threshold
- Extended-stay siren alerts
- Unattended object detection for bags, suitcases, bottles, laptops, and more
- Timestamped evidence screenshots
- Recent incident history with image previews
- Uploaded video, local video file, webcam, or IP camera support
- Adjustable detection thresholds from the dashboard sidebar

## How It Works

1. The Streamlit app receives a video from an upload, local file, webcam, or IP camera.
2. OpenCV reads and processes each frame.
3. YOLOv8 detects people and selected objects.
4. ByteTrack maintains tracking IDs across frames.
5. Detection rules determine whether a person entered or remained in the restricted zone.
6. The left-object tracker identifies objects that remain away from an owner.
7. Alerts are saved as screenshots and logged in `incidents.db`.

## Tech Stack

- Python 3.11
- YOLOv8 / Ultralytics
- ByteTrack
- OpenCV
- Streamlit
- NumPy
- Pandas
- SQLite

## Project Structure

```text
.
|-- app.py             # Streamlit dashboard and main detection loop
|-- detector.py        # People detection, tracking, and activity rules
|-- left_objects.py    # Unattended-object tracking and alerts
|-- alerts.py          # Evidence capture and alert sound generation
|-- database.py        # SQLite incident database
|-- config.py          # Paths and detection thresholds
|-- data/              # Uploaded and existing videos
|-- evidence/          # Saved alert screenshots
|-- incidents.db       # SQLite incident history
|-- requirements.txt   # Python dependencies
`-- run_app.bat        # Optional Windows launcher
```

## Requirements

- Windows, Linux, or macOS
- Python 3.10 or 3.11
- Internet access on the first run to download YOLO model weights
- A webcam or video source, if using live input

## Installation

Clone the repository and enter the project folder:

```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

Create and activate a virtual environment:

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

From the project directory, run:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

On Windows, the included launcher can also be used after creating a `venv` environment:

```bash
run_app.bat
```

## Using the Dashboard

1. Select **Uploaded video** or **Webcam / IP camera**.
2. Upload a video, choose an existing file, or enter a webcam index or camera URL.
3. Adjust the restricted zone using the left, right, top, and bottom sliders.
4. Set the loitering, siren, and unattended-object thresholds.
5. Enable or disable alert sounds.
6. Toggle **Run detection** to start processing.
7. Review current alerts and previous incidents in the dashboard.

For webcam input, enter `0` for the default camera. IP cameras can use an RTSP or
HTTP stream URL supported by OpenCV.

## Default Detection Rules

| Rule | Default threshold |
|---|---:|
| Restricted-zone entry | Immediate |
| Loitering alert | 20 seconds |
| Extended-stay siren | 25 seconds |
| Unattended object | 60 seconds |

Thresholds can be changed in the dashboard sidebar or updated in `config.py`.

## Detection Models

The application uses a YOLOv8 nano model by default. On the first run, Ultralytics
downloads the model weight file automatically.

Additional object detection settings are configured in `config.py`, including:

- Object classes
- Confidence threshold
- Object detection frequency
- Matching distance
- Owner distance
- Missing-object memory
- Object alert threshold

Lower the frame width or detection frequency if processing is slow on limited hardware.

## Database and Evidence

All incidents are stored in `incidents.db` in the project directory.

Each database record contains:

- Incident ID
- Real-world creation time
- Event type
- ByteTrack tracking ID
- Video time
- Evidence image path

Evidence images are saved in the `evidence/` folder.

Use **Clear incident history** in the dashboard to remove database records. This does
not delete existing evidence image files.

## Troubleshooting

### Streamlit cannot start

Reinstall dependencies in the active virtual environment:

```bash
python -m pip install -r requirements.txt
```

### YOLO model download fails

Ensure internet access is available for the first run and confirm that the required
weight file can be downloaded by Ultralytics.

### Webcam cannot be opened

- Try a different numeric camera index.
- Close applications currently using the webcam.
- Confirm the IP camera URL is accessible.

### Detection is slow

- Use the nano model.
- Lower `FRAME_WIDTH` in `config.py`.
- Process shorter or lower-resolution videos.
- Increase `OBJECT_EVERY_N_FRAMES` to run object detection less frequently.

## Privacy and Responsible Use

This project processes surveillance footage locally. When deployed in a real
environment, obtain appropriate consent, follow applicable privacy laws, restrict
database and video access, and use the alerts as a decision-support tool rather than
the sole basis for enforcement.

## License

This project does not currently include a license file. Add an OSI-approved license
before using or distributing the project publicly.
