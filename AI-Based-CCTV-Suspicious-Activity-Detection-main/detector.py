# ---------- Detection + Tracking + Activity Rules ----------
import cv2
from ultralytics import YOLO
from config import (MODEL_NAME, CONFIDENCE, OBJECT_CLASSES, OBJECT_CONFIDENCE,
                    OBJECT_EVERY_N_FRAMES)
from left_objects import LeftObjectTracker


class ActivityDetector:
    def __init__(self, loiter_seconds, sound_seconds, object_seconds):
        self.model = YOLO(MODEL_NAME)          # model 1: finds and tracks PEOPLE
        self.obj_model = YOLO(MODEL_NAME)      # model 2: finds OBJECTS (kept separate so it
                                               # cannot disturb the people tracker)
        self.loiter_seconds = loiter_seconds   # threshold, e.g. 20
        self.sound_seconds = sound_seconds     # alarm threshold, e.g. 25
        self.left_tracker = LeftObjectTracker(object_seconds)   # left-behind rule, e.g. 60
        self.frame_count = 0

        # Memory about each person, keyed by their tracking ID
        self.entry_time = {}         # id -> time the person entered the zone
        self.entry_alerted = set()   # ids already alerted for zone entry
        self.loiter_alerted = set()  # ids already alerted for loitering
        self.sound_alerted = set()   # ids already given the alarm sound

    def _detect_objects(self, frame):
        """Run YOLO for bags / suitcases / laptops. Returns [(name, (x1,y1,x2,y2)), ...]."""
        res = self.obj_model.predict(frame, classes=list(OBJECT_CLASSES.keys()),
                                     conf=OBJECT_CONFIDENCE, verbose=False)[0]
        found = []
        if len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.int().cpu().tolist()
            for box, cls in zip(boxes, classes):
                found.append((OBJECT_CLASSES[cls], tuple(float(v) for v in box)))
        return found

    def process(self, frame, zone_poly, now):
        """
        frame     : one video frame (BGR image)
        zone_poly : np.int32 array of polygon points (the restricted area)
        now       : current time in seconds
        Returns   : (annotated frame, list of (event_type, id))
        """
        events = []

        # --- People: detection + tracking in one call ---
        # classes=[0]  -> only detect 'person'
        # persist=True -> keep the same ID for a person across frames
        result = self.model.track(
            frame, persist=True, classes=[0], conf=CONFIDENCE,
            tracker="bytetrack.yaml", verbose=False
        )[0]

        # --- Objects: look for bags etc. every few frames, on the CLEAN frame ---
        detections = None    # None = "skipped this frame"
        if self.frame_count % OBJECT_EVERY_N_FRAMES == 0:
            detections = self._detect_objects(frame)
        self.frame_count += 1

        # Boxes of every person (the object rule needs them to know who is nearby)
        person_boxes = []
        if len(result.boxes):
            person_boxes = [tuple(float(v) for v in b)
                            for b in result.boxes.xyxy.cpu().numpy()]

        # Draw the restricted zone in red
        cv2.polylines(frame, [zone_poly], True, (0, 0, 255), 2)

        inside_now = set()   # people inside the zone in THIS frame

        # result.boxes.id is None when nobody is tracked
        if result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy()    # box corners
            ids = result.boxes.id.int().cpu().tolist() # tracking IDs

            for (x1, y1, x2, y2), tid in zip(boxes, ids):
                # Use the FEET (bottom-centre of the box) as the person's
                # position on the floor. This is more accurate than the box centre.
                foot = (float((x1 + x2) / 2), float(y2))

                # --- Rule 1: is the person inside the restricted zone? ---
                inside = cv2.pointPolygonTest(zone_poly, foot, False) >= 0
                color = (0, 255, 0)   # green = normal

                if inside:
                    inside_now.add(tid)
                    color = (0, 165, 255)   # orange = inside zone

                    # First frame inside -> start this person's timer
                    if tid not in self.entry_time:
                        self.entry_time[tid] = now

                    # Alert only once per person for zone entry
                    if tid not in self.entry_alerted:
                        self.entry_alerted.add(tid)
                        events.append(("Restricted zone entry", tid))

                    # --- Rule 2: still inside after N seconds -> loitering ---
                    dwell = now - self.entry_time[tid]
                    cv2.putText(frame, f"{dwell:.0f}s", (int(x1), int(y2) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    if dwell >= self.loiter_seconds and tid not in self.loiter_alerted:
                        self.loiter_alerted.add(tid)
                        events.append(("Loitering in restricted zone", tid))
                        color = (0, 0, 255)   # red = suspicious

                    # --- Rule 3: inside for the sound threshold -> alarm ---
                    if dwell >= self.sound_seconds and tid not in self.sound_alerted:
                        self.sound_alerted.add(tid)
                        events.append(("Extended stay alarm", tid))
                        color = (0, 0, 255)   # red

                # Draw the box, ID label, and foot point
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.circle(frame, (int(foot[0]), int(foot[1])), 4, color, -1)

        # --- Rule 4: object left behind (bag alone in the zone for N seconds) ---
        for obj in self.left_tracker.update(detections, person_boxes, zone_poly, now):
            events.append(("Object left behind", obj.oid))
        self.left_tracker.draw(frame, now)      # triangles + labels for objects

        # People who left the zone: reset their timer so a re-entry starts fresh
        for tid in list(self.entry_time):
            if tid not in inside_now:
                del self.entry_time[tid]
                self.loiter_alerted.discard(tid)
                self.sound_alerted.discard(tid)

        return frame, events
