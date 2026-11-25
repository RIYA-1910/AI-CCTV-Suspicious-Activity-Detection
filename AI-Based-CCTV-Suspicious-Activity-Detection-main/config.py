# ---------- Central settings: change values here, not in the logic ----------
import os

# Folder where this file lives (works on any computer)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EVIDENCE_DIR = os.path.join(BASE_DIR, "evidence")   # saved alert screenshots
DATA_DIR = os.path.join(BASE_DIR, "data")           # test videos
SOUND_DIR = os.path.join(BASE_DIR, "sounds")        # alarm WAV files (created automatically)
DB_PATH = os.path.join(BASE_DIR, "incidents.db")    # SQLite database file

MODEL_NAME = "yolov8n.pt"    # people model. "n" = nano: smallest and fastest.
CONFIDENCE = 0.4             # ignore person detections below 40% confidence
FRAME_WIDTH = 1280           # wider frame keeps more detail from 4K source videos

# Default thresholds (can be changed on the dashboard sidebar)
DEFAULT_LOITER_SECONDS = 20  # loitering alert after this many seconds in the zone
DEFAULT_SOUND_SECONDS = 25   # siren sound after this many seconds in the zone
DEFAULT_OBJECT_SECONDS = 60  # object left alone this long -> alert + beep sound

# ---------- Object left-behind settings ----------
# Objects are small, so they get their own, stronger model and settings.
OBJECT_MODEL_NAME = "yolov8s.pt"   # better at finding bags than "n". Downloads once (~22 MB).
                                   # If the app is too slow, change it to "yolov8n.pt".
OBJECT_IMGSZ = 1280                # larger input = better detection on 4K videos resized to 960px
OBJECT_CONFIDENCE = 0.15           # low: backpack in this video detected at conf=0.16
OBJECT_EVERY_N_FRAMES = 5          # look for objects every 5th frame (they don't move)

# COCO class id -> name. Add or remove objects here.
OBJECT_CLASSES = {24: "backpack", 25: "umbrella", 26: "handbag", 28: "suitcase",
                  39: "bottle", 41: "cup", 63: "laptop", 67: "cell phone", 73: "book"}

OBJECT_MATCH_DISTANCE = 80         # pixels: same object if its centre moved less than this
OBJECT_OWNER_DISTANCE = 100        # pixels: a person closer than this is "with" the object
OBJECT_OWNER_MEMORY_SECONDS = 30   # a person who stood near the spot in the last 30 s counts as owner
OBJECT_MISSING_SECONDS = 15        # keep drawing the last known triangle this long when YOLO loses it
OBJECT_LOST_MEMORY_SECONDS = 300   # remember a lost object (and its timer) this long, in case it is found again
OBJECT_REQUIRE_OWNER = True        # True: only alert if a person was seen with the object

# Create the folders if they don't exist
os.makedirs(EVIDENCE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SOUND_DIR, exist_ok=True)
