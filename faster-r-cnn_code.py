import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import mediapipe as mp

# Setup
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Faster R-CNN model
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load("training-ssd/faster_rcnn_handball.pth", map_location=device))
model.eval().to(device)

# Video IO
cap = cv2.VideoCapture("input_videos/jacob.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_videos/output_frcnn.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Label map — check your dataset export here
class_names = {0: "__background__", 1: "goalpost", 2: "handball"}

# MediaPipe pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Torch tensor
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Faster R-CNN prediction
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    goalpost_box = None
    ball_x, ball_y = None, None

    # Parse predictions
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cls_name = class_names[label.item()]

        if cls_name == "goalpost":
            goalpost_box = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        elif cls_name == "handball":
            ball_x, ball_y = cx, cy
            cv2.circle(frame, (ball_x, ball_y), 6, (0, 255, 0), -1)

    # Shot classification
    if goalpost_box and ball_x is not None:
        gx1, gy1, gx2, gy2 = goalpost_box
        gw, gh = gx2 - gx1, gy2 - gy1
        gcx, gcy = (gx1 + gx2) // 2, (gy1 + gy2) // 2

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
        
    # Outline goalpost regions with rectangles
    # Outline goalpost regions with non-overlapping rectangles
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
        # Center (vertical)
        cv2.rectangle(frame, (center_x1, gy1), (center_x2, gy2), (0, 0, 255), 2)
        # Bottom-left
        cv2.rectangle(frame, (gx1, gy1 + (2 * gh) // 3), (center_x1, gy2), (0, 0, 255), 2)
        # Bottom-right
        cv2.rectangle(frame, (center_x2, gy1 + (2 * gh) // 3), (gx2, gy2), (0, 0, 255), 2)

    # MediaPipe Pose estimation
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Convert normalized coords to pixels
        h, w, _ = frame.shape
        right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        elbow_px = int(right_elbow.x * w), int(right_elbow.y * h)
        shoulder_px = int(right_shoulder.x * w), int(right_shoulder.y * h)

        # Visualize keypoints
        cv2.circle(frame, elbow_px, 6, (0, 255, 255), -1)
        cv2.circle(frame, shoulder_px, 6, (255, 255, 0), -1)

        # Check throwing form
        if right_elbow.y < right_shoulder.y:  # elbow higher (smaller y)
            cv2.putText(frame, "Good Throwing Form", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Faster R-CNN + Pose", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
