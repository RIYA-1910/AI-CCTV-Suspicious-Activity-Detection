# ---------- Alert System: screenshot + timestamp + DB record + sounds ----------
import io
import os
import wave
import cv2
import numpy as np
from datetime import datetime
from config import EVIDENCE_DIR
import database


def raise_alert(frame, event_type, track_id, video_time):
    """Save an evidence frame and create an incident record."""
    # Build a unique filename like 20261024_013000_restricted_zone_entry_id3.jpg
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = event_type.lower().replace(" ", "_")
    filename = f"{stamp}_{safe_type}_id{track_id}.jpg"
    path = os.path.join(EVIDENCE_DIR, filename)

    # Copy the frame so we don't change the one shown on the dashboard,
    # then write the event name and time on it as proof
    evidence = frame.copy()
    label = f"{event_type} | ID {track_id} | {datetime.now():%Y-%m-%d %H:%M:%S}"
    cv2.putText(evidence, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.imwrite(path, evidence)

    # Store the incident in the database
    database.add_incident(event_type, track_id, video_time, path)
    return {"event_type": event_type, "track_id": track_id, "image_path": path}


def _to_wav(samples, sample_rate):
    """Pack float samples (range -1..1) into 16-bit mono WAV bytes."""
    data = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)          # mono
        w.setsampwidth(2)          # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(data.tobytes())
    return buf.getvalue()


def make_alarm_wav(duration=2.0, sample_rate=22050):
    """Sound 1 (long stay in zone): a two-tone siren, 880 Hz / 660 Hz."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Switch between two pitches every 0.25 s to sound like a siren
    freq = np.where((t // 0.25) % 2 == 0, 880, 660)
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate   # keeps the wave smooth
    return _to_wav(0.6 * np.sin(phase), sample_rate)


def make_object_alarm_wav(extra=0.0, sample_rate=22050):
    """Sound 2 (object left behind): three sharp high beeps, clearly
    different from the siren. 'extra' adds a little silence at the end so every
    clip has different bytes, which makes the browser play it again."""
    beep, gap, fade = 0.18, 0.12, 0.01
    n = int(sample_rate * beep)
    t = np.arange(n) / sample_rate
    tone = 0.6 * np.sin(2 * np.pi * 1200 * t)            # 1200 Hz beep
    f = int(sample_rate * fade)                          # short fade in/out: no clicks
    envelope = np.ones(n)
    envelope[:f] = np.linspace(0, 1, f)
    envelope[-f:] = np.linspace(1, 0, f)
    silence = np.zeros(int(sample_rate * gap))
    pattern = np.concatenate([tone * envelope, silence] * 3)
    pattern = np.concatenate([pattern, np.zeros(int(sample_rate * extra))])
    return _to_wav(pattern, sample_rate)
