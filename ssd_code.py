import cv2
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import ssd300_vgg16
import mediapipe as mp

# ----------------------------
# Settings
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3  # background + goalpost + handball
CONF_THRESHOLD = 0.3  # minimum confidence to show box

# Paths
VIDEO_INPUT = "input_videos/jacob.mp4"
VIDEO_OUTPUT = "output_videos/output_ssd.mp4"
MODEL_PATH = "training-ssd/ssd_trained_handball.pth"

# ----------------------------
# Load model
# ----------------------------
model = ssd300_vgg16(weights=None, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# ----------------------------
# Label map
# ----------------------------
class_names = {0: "__background__", 1: "handball", 2: "post"}

# ----------------------------
# MediaPipe pose setup
# ----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# ----------------------------
# Video IO setup
# ----------------------------
cap = cv2.VideoCapture(VIDEO_INPUT)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    VIDEO_OUTPUT,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# ----------------------------
# Main loop
# ----------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(DEVICE)

    # SSD inference
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    print(outputs['labels'])
    print(outputs['scores'])
    print(outputs['boxes'])

    goalpost_box = None
    ball_x, ball_y = None, None

    # Parse predictions
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        cls_name = class_names[label.item()]

        if cls_name == "post":
            goalpost_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{cls_name}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        elif cls_name == "handball":
            ball_x, ball_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (ball_x, ball_y), 6, (0, 255, 0), -1)
            cv2.putText(frame, f"{cls_name}: {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ----------------------------
    # Shot classification (goal target)
    # ----------------------------
    if goalpost_box and ball_x is not None:
        gx1, gy1, gx2, gy2 = goalpost_box
        gw, gh = gx2 - gx1, gy2 - gy1
        gcx = (gx1 + gx2) // 2

        if ball_x < gx1 or ball_x > gx2:
            position_text = "Out of Target"
        elif ball_y < gy1 + gh // 3:
            position_text = "Top Left" if ball_x < gcx else "Top Right"
        elif ball_y > gy1 + (2 * gh) // 3:
            position_text = "Bottom Left" if ball_x < gcx else "Bottom Right"
        else:
            position_text = "Center"

        cv2.putText(frame, position_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    if goalpost_box:
        gx1, gy1, gx2, gy2 = goalpost_box
        gw, gh = gx2 - gx1, gy2 - gy1
        gcx = (gx1 + gx2) // 2

        # Define center vertical strip width
        center_width = gw // 3
        center_x1 = gx1 + (gw - center_width) // 2
        center_x2 = center_x1 + center_width

        # Top-left
        cv2.rectangle(frame, (gx1, gy1), (center_x1, gy1 + gh // 3), (0, 0, 255), 2)
        # Top-right
        cv2.rectangle(frame, (center_x2, gy1), (gx2, gy1 + gh // 3), (0, 0, 255), 2)
        # Center vertical
        cv2.rectangle(frame, (center_x1, gy1), (center_x2, gy2), (0, 0, 255), 2)
        # Bottom-left
        cv2.rectangle(frame, (gx1, gy1 + (2 * gh) // 3), (center_x1, gy2), (0, 0, 255), 2)
        # Bottom-right
        cv2.rectangle(frame, (center_x2, gy1 + (2 * gh) // 3), (gx2, gy2), (0, 0, 255), 2)

    # ----------------------------
    # MediaPipe Pose
    # ----------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        h, w, _ = frame.shape
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        elbow_px = int(right_elbow.x * w), int(right_elbow.y * h)
        shoulder_px = int(right_shoulder.x * w), int(right_shoulder.y * h)

        cv2.circle(frame, elbow_px, 6, (0, 255, 255), -1)
        cv2.circle(frame, shoulder_px, 6, (255, 255, 0), -1)

        # Throwing form
        if right_elbow.y < right_shoulder.y:
            cv2.putText(frame, "Good Throwing Form", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ----------------------------
    # Display & save
    # ----------------------------
    cv2.imshow("SSD Handball Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
