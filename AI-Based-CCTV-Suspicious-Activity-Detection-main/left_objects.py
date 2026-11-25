# ---------- Object left-behind logic + triangle drawing ----------
# This file does not need YOLO: it receives detected object boxes and person
# boxes and decides when an object has been left alone for too long.
import math
from collections import deque

import cv2
import numpy as np
from config import (OBJECT_MATCH_DISTANCE, OBJECT_OWNER_DISTANCE,
                    OBJECT_OWNER_MEMORY_SECONDS, OBJECT_MISSING_SECONDS,
                    OBJECT_LOST_MEMORY_SECONDS, OBJECT_REQUIRE_OWNER)


class TrackedObject:
    """One bag / suitcase / laptop that we are following."""

    def __init__(self, oid, name, box, now):
        self.oid = oid                  # object id (starts at 1001, never clashes with person ids)
        self.name = name                # e.g. "backpack"
        self.box = box                  # (x1, y1, x2, y2) - last known position
        self.last_seen = now            # last time YOLO saw it
        self.lost_at = None             # when we gave up drawing it
        self.had_owner = False          # was a person ever close to it?
        self.unattended_since = None    # time when the last person walked away
        self.alerted = False            # already raised the left-behind alert?
        self.in_zone = False            # filled in every frame (used for drawing)
        self.attended = False           # filled in every frame (used for drawing)

    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def _distance_to_box(px, py, box):
    """Distance from a point to a rectangle (0 if the point is inside it)."""
    x1, y1, x2, y2 = box
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.hypot(dx, dy)


class LeftObjectTracker:
    def __init__(self, left_seconds):
        self.left_seconds = left_seconds   # e.g. 60
        self.objects = {}                  # objects we can see now (or saw a moment ago)
        self.lost = {}                     # objects YOLO lost; remembered so they can come back
        self.next_id = 1001
        self.recent_people = deque()       # (time, [person boxes]) from the last few seconds
        self.missing_seconds = min(OBJECT_MISSING_SECONDS, left_seconds / 2)

    # ---------- helpers ----------
    @staticmethod
    def _nearest(pool, cx, cy, used, max_dist):
        """Id of the object in 'pool' whose centre is closest to (cx,cy),
        only if it is within max_dist pixels."""
        best, best_d = None, max_dist
        for oid, obj in pool.items():
            if oid in used:
                continue
            ox, oy = obj.center
            d = math.hypot(cx - ox, cy - oy)
            if d < best_d:
                best, best_d = oid, d
        return best

    def _last_person_near(self, cx, cy):
        """Most recent time a person was close to this spot (None if nobody was)."""
        for t, boxes in reversed(self.recent_people):
            if any(_distance_to_box(cx, cy, b) <= OBJECT_OWNER_DISTANCE for b in boxes):
                return t
        return None

    # ---------- main update ----------
    def update(self, detections, person_boxes, zone_poly, now):
        """
        detections   : list of (name, (x1, y1, x2, y2)) found in this frame,
                       or None if object detection was skipped for this frame
        person_boxes : list of (x1, y1, x2, y2) for every person in this frame
        zone_poly    : np.int32 polygon of the monitored zone
        now          : current time in seconds
        Returns the objects that JUST crossed the left-behind time limit.
        """
        # 0. Remember where people have been during the last few seconds
        if person_boxes:
            self.recent_people.append((now, person_boxes))
        while self.recent_people and now - self.recent_people[0][0] > OBJECT_OWNER_MEMORY_SECONDS:
            self.recent_people.popleft()

        # 1. Match this frame's detections to objects we already know.
        #    Use a GENEROUS match distance so a moving bag keeps its ID even
        #    when YOLO skips several frames (objects are detected every Nth frame).
        if detections is not None:
            # Sort detections by confidence proxy: larger box = more confident
            detections = sorted(detections, key=lambda d: -(d[1][2]-d[1][0])*(d[1][3]-d[1][1]))
            used = set()
            for name, box in detections:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

                # A person box that completely contains this detection is likely a false
                # positive (e.g. YOLO calling a person's torso a "cell phone").
                # Skip it.
                is_inside_person = False
                for pb in person_boxes:
                    px1, py1, px2, py2 = pb
                    if px1 <= cx <= px2 and py1 <= cy <= py2:
                        is_inside_person = True
                        break
                if is_inside_person:
                    continue

                # Try to match to a currently-visible object first
                best = self._nearest(self.objects, cx, cy, used, OBJECT_MATCH_DISTANCE)
                if best is None:
                    # Then try the lost list (wider search: the object may have moved)
                    best = self._nearest(self.lost, cx, cy, used, OBJECT_MATCH_DISTANCE * 2)
                    if best is not None:
                        self.objects[best] = self.lost.pop(best)  # restore with timer intact

                if best is None:
                    # Genuinely new object
                    best = self.next_id
                    self.next_id += 1
                    obj = TrackedObject(best, name, box, now)
                    # If a person stood near here recently they were likely the owner
                    owner_time = self._last_person_near(cx, cy)
                    if owner_time is not None:
                        obj.had_owner = True
                        obj.unattended_since = owner_time
                    self.objects[best] = obj

                # Always update position so the triangle follows the object
                obj = self.objects[best]
                obj.name, obj.box, obj.last_seen = name, box, now
                used.add(best)

        # 2. Objects YOLO has not seen for a while move to the 'lost' list
        for oid in list(self.objects):
            if now - self.objects[oid].last_seen > self.missing_seconds:
                obj = self.objects.pop(oid)
                obj.lost_at = now
                self.lost[oid] = obj
        for oid in list(self.lost):
            if now - self.lost[oid].lost_at > OBJECT_LOST_MEMORY_SECONDS:
                del self.lost[oid]

        # 3. Apply the alert rule to every currently-visible object
        newly_left = []
        for obj in self.objects.values():
            cx, cy = obj.center
            obj.in_zone = cv2.pointPolygonTest(zone_poly, (float(cx), float(cy)), False) >= 0
            obj.attended = any(_distance_to_box(cx, cy, pb) <= OBJECT_OWNER_DISTANCE
                               for pb in person_boxes)

            if obj.attended:
                obj.had_owner = True
                obj.unattended_since = None
                obj.alerted = False
                continue

            if not obj.in_zone:
                obj.unattended_since = None
                continue

            if obj.unattended_since is None:
                obj.unattended_since = now

            waited = now - obj.unattended_since
            if ((obj.had_owner or not OBJECT_REQUIRE_OWNER)
                    and not obj.alerted and waited >= self.left_seconds):
                obj.alerted = True
                newly_left.append(obj)
        return newly_left

    def draw(self, frame, now):
        """Draw a triangle around each tracked object and a status label.

        Visible objects  -> triangle follows the live detection box.
        Temporarily-lost objects (YOLO missed them for a moment) -> draw at last
        known position ONLY if they were in the zone, so ghost triangles don't
        appear all over the frame for objects that have simply walked away.
        """
        # Draw lost objects first (underneath), then active objects on top
        for obj in list(self.lost.values()):
            # Only keep drawing a lost object that was inside the zone.
            # This avoids ghost triangles for carried bags that left the scene.
            if not obj.in_zone:
                continue
            self._draw_one(frame, obj, now, ghost=True)

        for obj in self.objects.values():
            self._draw_one(frame, obj, now, ghost=False)

        return frame

    def _draw_one(self, frame, obj, now, ghost=False):
        x1, y1, x2, y2 = obj.box
        w = (x2 - x1) / 2 + 8    # half-width of the base (8 px margin)
        cx = (x1 + x2) / 2

        # Triangle: apex above the box, base below it
        tri = np.array([[cx,     y1 - 10],
                        [cx - w, y2 +  8],
                        [cx + w, y2 +  8]], dtype=np.int32)

        name = f"{obj.name} #{obj.oid}"
        if obj.alerted:
            color, label = (0, 0, 255), f"{name} LEFT BEHIND"       # red
        elif ghost:
            color, label = (180, 180, 180), f"{name} (tracking...)" # grey - temporarily lost
        elif not obj.in_zone:
            color, label = (180, 180, 180), name                    # grey - outside zone
        elif obj.attended:
            color, label = (0, 255, 0), name                        # green - owner present
        elif obj.unattended_since is not None and (obj.had_owner or not OBJECT_REQUIRE_OWNER):
            secs = now - obj.unattended_since
            color, label = (0, 165, 255), f"{name} alone {secs:.0f}s"  # orange - timer running
        else:
            color, label = (0, 255, 255), name                      # yellow - no owner seen yet

        cv2.polylines(frame, [tri], True, color, 2)
        cv2.putText(frame, label, (int(x1), int(y2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
