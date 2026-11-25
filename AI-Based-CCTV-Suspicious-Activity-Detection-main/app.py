# ---------- Streamlit Dashboard + Video Input + main loop ----------
import os
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st

import database
from alerts import raise_alert, make_alarm_wav, make_object_alarm_wav
from config import (DATA_DIR, FRAME_WIDTH, DEFAULT_LOITER_SECONDS,
                    DEFAULT_SOUND_SECONDS, DEFAULT_OBJECT_SECONDS)
from detector import ActivityDetector

st.set_page_config(page_title="CCTV Suspicious Activity Detection", layout="wide")
st.title("AI CCTV Suspicious Activity Detection")
database.init_db()   # make sure the incidents table exists


def resize(frame):
    """Shrink the frame to FRAME_WIDTH so processing is faster."""
    h, w = frame.shape[:2]
    return cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))


def zone_from_percent(frame, x1, y1, x2, y2):
    """Convert slider percentages into a rectangle of pixel points."""
    h, w = frame.shape[:2]
    return np.array([[int(w*x1/100), int(h*y1/100)],
                     [int(w*x2/100), int(h*y1/100)],
                     [int(w*x2/100), int(h*y2/100)],
                     [int(w*x1/100), int(h*y2/100)]], dtype=np.int32)


# ===== SIDEBAR: Module 1 (Video Input) + Module 4 (Restricted Zone) =====
st.sidebar.header("1. Video source")
source_type = st.sidebar.radio("Source", ["Uploaded video", "Webcam / IP camera"])

source = None
if source_type == "Uploaded video":
    up = st.sidebar.file_uploader("Upload CCTV video", type=["mp4", "avi", "mov", "mkv"])
    if up is not None:
        # Save the uploaded file into data/ so OpenCV can read it from disk
        source = os.path.join(DATA_DIR, "uploaded_" + up.name)
        with open(source, "wb") as f:
            f.write(up.getbuffer())
    else:
        # Otherwise let the user pick a video already in data/
        existing = [f for f in os.listdir(DATA_DIR)
                    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        choice = st.sidebar.selectbox("...or pick a file from data/", ["-"] + existing)
        if choice != "-":
            source = os.path.join(DATA_DIR, choice)
else:
    # "0" = default webcam; or paste an rtsp:// / http:// IP camera URL
    cam = st.sidebar.text_input("Webcam index (0) or IP camera URL", "0")
    source = int(cam) if cam.isdigit() else cam

st.sidebar.header("2. Restricted zone (% of frame)")
zx1 = st.sidebar.slider("Left", 0, 100, 30)
zx2 = st.sidebar.slider("Right", 0, 100, 70)
zy1 = st.sidebar.slider("Top", 0, 100, 40)
zy2 = st.sidebar.slider("Bottom", 0, 100, 95)

st.sidebar.header("3. Rules")
loiter_seconds = st.sidebar.number_input("Loitering threshold (seconds)", 1, 600,
                                         DEFAULT_LOITER_SECONDS)
sound_seconds = st.sidebar.number_input("Alert sound threshold (seconds)", 1, 600,
                                        DEFAULT_SOUND_SECONDS)
object_seconds = st.sidebar.number_input("Object left-behind threshold (seconds)", 1, 3600,
                                         DEFAULT_OBJECT_SECONDS)

st.sidebar.header("4. Sound")
sound_on = st.sidebar.checkbox("Enable alert sound", value=True)
browser_sound = st.sidebar.checkbox("Also play sound in the browser", value=False)
test_sound = st.sidebar.button("Test alert sound (siren)")
test_object_sound = st.sidebar.button("Test object alert sound (3 beeps)")

run = st.sidebar.toggle("Run detection")
if st.sidebar.button("Clear incident history"):
    database.clear_incidents()

# ===== MAIN LAYOUT =====
left, right = st.columns([3, 2])
video_slot = left.empty()     # placeholder that we keep overwriting with new frames
status_slot = left.empty()
right.subheader("Current alerts")
alert_slot = right.empty()

sound_slot = right.empty()      # browser audio player (used only if enabled / needed)


def play_sound(kind, count=0):
    """kind = 'siren' or 'object'.
    Always plays in the browser (autoplay=True) AND through the PC speakers.
    The WAV is regenerated with a unique length each time so the browser
    treats it as a brand-new clip and plays it again."""
    # Browser: unique clip so autoplay fires every time
    extra = 0.001 * (count % 1000)   # 0–1 ms of silence appended; makes bytes unique
    if kind == "siren":
        wav = make_alarm_wav(2.0 + extra)
    else:
        wav = make_object_alarm_wav(extra)
    sound_slot.audio(wav, format="audio/wav", autoplay=True)


if sound_on and test_sound:
    play_sound("siren", 0)
if sound_on and test_object_sound:
    play_sound("object", 0)

# ===== Preview the zone on the first frame when not running =====
if source is not None and not run:
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if ok:
        frame = resize(frame)
        poly = zone_from_percent(frame, zx1, zy1, zx2, zy2)
        cv2.polylines(frame, [poly], True, (0, 0, 255), 3)
        video_slot.image(frame, channels="BGR",
                         caption="Zone preview - adjust sliders, then toggle Run")
    else:
        status_slot.error("Could not read the video source.")

# ===== MAIN LOOP: this is the pipeline from your diagram =====
if run and source is not None:
    cap = cv2.VideoCapture(source)
    is_file = isinstance(source, str) and os.path.isfile(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector = ActivityDetector(loiter_seconds, sound_seconds, object_seconds)
    alarm_count = 0
    session_alerts = []
    frame_idx = 0
    start_wall = time.time()        # used as the clock for live cameras

    while cap.isOpened():
        ok, frame = cap.read()          # 1. get a video frame
        if not ok:
            status_slot.info("Video finished.")
            break

        frame = resize(frame)
        # Files: time comes from the frame number, so the rules are correct
        # no matter how fast or slow your computer processes.
        # Live cameras: use the real clock.
        now = frame_idx / fps if is_file else time.time() - start_wall
        frame_idx += 1

        zone_poly = zone_from_percent(frame, zx1, zy1, zx2, zy2)

        # 2-5. YOLO detection -> tracking -> activity rules
        annotated, events = detector.process(frame, zone_poly, now)

        # 6. Suspicious? YES -> alert + screenshot + timestamp + database + sound
        for event_type, tid in events:
            info = raise_alert(annotated, event_type, tid, now)
            session_alerts.insert(0, info)

            if sound_on and event_type == "Extended stay alarm":
                alarm_count += 1
                play_sound("siren", alarm_count)
            elif sound_on and event_type == "Object left behind":
                alarm_count += 1
                play_sound("object", alarm_count)

        video_slot.image(annotated, channels="BGR")
        n_objects = len(detector.left_tracker.objects)
        status_slot.caption(f"Video time: {now:.1f}s | Frame {frame_idx} | "
                            f"Objects tracked: {n_objects}")

        if events:
            with alert_slot.container():
                for a in session_alerts[:3]:
                    st.error(f"ALERT: {a['event_type']} (ID {a['track_id']})")
                    st.image(a["image_path"])

    cap.release()

# ===== Module 7: Incident history (dashboard) =====
st.divider()
st.subheader("Previous incidents")
rows = database.get_incidents()
if rows:
    df = pd.DataFrame(rows)[["id", "created_at", "event_type", "track_id", "video_time"]]
    st.dataframe(df, hide_index=True)
    st.write("**Evidence images (latest 6)**")
    cols = st.columns(3)
    for i, r in enumerate(rows[:6]):
        with cols[i % 3]:
            if os.path.exists(r["image_path"]):
                st.image(r["image_path"],
                         caption=f"#{r['id']} {r['event_type']} @ {r['created_at']}")
else:
    st.info("No incidents recorded yet.")
