import cv2
from ultralytics import YOLO
import time

# ---------------- CONFIG ----------------
VIDEO_PATH = r"D:\aiproject\test_video_converted1.mp4"
MODEL_NAME = "yolov8n.pt"
KNOWN_WIDTH = 1.8
FOCAL_LENGTH = 555.56
EMERGENCY_CLASSES = ["truck", "bus"]
RELEVANT_VEHICLES = ["car", "bus", "truck", "motorbike"]
EMERGENCY_DISTANCE_THRESHOLD = 20

# Traffic light colors
RED = (0,0,255)
YELLOW = (0,255,255)
GREEN = (0,255,0)
COUNTDOWN_COLOR = (255,0,0)  # Blue

# Traffic light timing
RED_TIME = 5
GREEN_TIME = 5
YELLOW_TIME = 2
EMERGENCY_GRACE_TIME = 2  # yellow before green

# ---------------- LANES CONFIG ----------------
lanes = {
    "straight": (100, 200, 300, 400),
    "left": (310, 200, 510, 400),
    "right": (520, 200, 720, 400),
    "opposite": (730, 200, 930, 400)
}

traffic_light_positions = {
    "straight": (180, 150),
    "left": (420, 150),
    "right": (640, 150),
    "opposite": (860, 150)
}

traffic_lights = {
    lane: {"state": "RED", "timer_start": time.time(), "timer_duration": RED_TIME,
           "emergency_override": False, "yellow_before_emergency": False, "post_emergency": False}
    for lane in lanes
}

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)
if not cap.isOpened():
    print(f"Error: Could not open video {VIDEO_PATH}")
    exit()

# ---------------- HELPER FUNCTION ----------------
def get_lane(cx, cy):
    for name, (x1, y1, x2, y2) in lanes.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return name
    return None

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    emergency_lanes = set()

    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2
        class_name = model.names[int(cls)]
        distance = round((KNOWN_WIDTH * FOCAL_LENGTH) / (x2 - x1), 2)

        lane = get_lane(cx, cy)
        if lane and class_name in RELEVANT_VEHICLES:
            # Draw lane rectangle for visualization
            cv2.rectangle(frame, (lanes[lane][0], lanes[lane][1]), (lanes[lane][2], lanes[lane][3]), (200,200,200), 1)

            # Draw bounding box only for emergency vehicles
            if class_name in EMERGENCY_CLASSES and distance <= EMERGENCY_DISTANCE_THRESHOLD:
                emergency_lanes.add(lane)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)
                cv2.putText(frame, f"EMERGENCY ({distance}m)", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    # ---------------- UPDATE TRAFFIC LIGHTS ----------------
    current_time = time.time()
    for lane_name, light in traffic_lights.items():
        elapsed = current_time - light["timer_start"]

        # Emergency vehicle detected
        if lane_name in emergency_lanes:
            if not light["emergency_override"] and not light["yellow_before_emergency"]:
                light["state"] = "YELLOW"
                light["timer_start"] = current_time
                light["timer_duration"] = EMERGENCY_GRACE_TIME
                light["yellow_before_emergency"] = True

        # Transition from yellow to green for emergency
        if light["yellow_before_emergency"] and elapsed >= light["timer_duration"]:
            light["state"] = "GREEN"
            light["timer_start"] = current_time
            light["timer_duration"] = GREEN_TIME  # now track elapsed green
            light["yellow_before_emergency"] = False
            light["emergency_override"] = True

        # Emergency green finished
        if light["emergency_override"] and lane_name not in emergency_lanes:
            # Extend green for normal duration before resuming cycle
            if not light["post_emergency"]:
                light["state"] = "GREEN"
                light["timer_start"] = current_time
                light["timer_duration"] = GREEN_TIME
                light["post_emergency"] = True
            elif elapsed >= light["timer_duration"]:
                light["state"] = "YELLOW"
                light["timer_start"] = current_time
                light["timer_duration"] = YELLOW_TIME
                light["emergency_override"] = False
                light["post_emergency"] = False

        # Normal traffic light cycle (if not emergency)
        if not light["emergency_override"] and not light["yellow_before_emergency"] and not light["post_emergency"]:
            if elapsed >= light["timer_duration"]:
                if light["state"] == "RED":
                    light["state"] = "GREEN"
                    light["timer_duration"] = GREEN_TIME
                elif light["state"] == "GREEN":
                    light["state"] = "YELLOW"
                    light["timer_duration"] = YELLOW_TIME
                elif light["state"] == "YELLOW":
                    light["state"] = "RED"
                    light["timer_duration"] = RED_TIME
                light["timer_start"] = current_time

    # ---------------- DRAW TRAFFIC LIGHTS ----------------
    for lane_name, light in traffic_lights.items():
        pos = traffic_light_positions[lane_name]
        color = GREEN if light["state"]=="GREEN" else YELLOW if light["state"]=="YELLOW" else RED
        elapsed = int(current_time - light["timer_start"])
        # Countdown display
        if light["emergency_override"]:
            countdown = f"{elapsed}s"  # elapsed green for emergency
        elif light["post_emergency"]:
            remaining = max(0, int(light["timer_duration"] - elapsed))
            countdown = f"{remaining}s"
        else:
            remaining = max(0, int(light["timer_duration"] - elapsed))
            countdown = f"{remaining}s"

        cv2.circle(frame, pos, 25, color, -1)
        cv2.putText(frame, countdown, (pos[0]-30, pos[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COUNTDOWN_COLOR, 2)

    cv2.imshow("Emergency Vehicle Traffic Control", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
