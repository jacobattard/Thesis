import cv2
from ultralytics import YOLO
import mediapipe as mp
import math

def load_model(model_path):
    model = YOLO(model_path)
    class_names = model.names
    post_idx = next((idx for idx, name in class_names.items() if name == "post"), None)
    ball_idx = next((idx for idx, name in class_names.items() if name == "handball"), None)
    if post_idx is None or ball_idx is None:
        raise ValueError("Error: 'post' or 'handball' class not found in the model!")
    return model, post_idx, ball_idx


def setup_video_io(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return cap, out


def draw_bounding_box(frame, box, color=(0, 0, 255)):
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def classify_shot(ball_x, ball_y, gx, gy, gw, gh):
    left = gx - gw // 2
    right = gx + gw // 2
    top = gy - gh // 2
    bottom = gy + gh // 2

    third_w = gw // 3
    third_h = gh // 3

    # Ignore out-of-goal shots
    if ball_x < left or ball_x > right or ball_y < top or ball_y > bottom:
        return "Out of Target"

    # Get column and row of the ball position
    col = int((ball_x - left) // third_w)
    row = int((ball_y - top) // third_h)

    # Map position
    if row == 0:
        return ["Top Left", "Top Center", "Top Right"][col]
    elif row == 1:
        return ["Left", "Center", "Right"][col]
    else:
        return ["Bottom Left", "Bottom Center", "Bottom Right"][col]


def process_frame(frame, results, post_idx, ball_idx):
    ball_x, ball_y = None, None
    goalpost_box = None

    for r in results:
        for box, cls in zip(r.boxes.xywh, r.boxes.cls):
            x, y, w, h = map(int, box.tolist())

            if int(cls) == post_idx:
                goalpost_box = (x, y, w, h)
                # Draw the full post bounding box (optional)
                cv2.rectangle(frame, (x - w//2, y - h//2), (x + w//2, y + h//2), (255, 255, 0), 2)

            elif int(cls) == ball_idx:
                ball_x, ball_y = x, y
                cv2.circle(frame, (ball_x, ball_y), 7, (0, 255, 0), -1)

    # If both ball and goalpost are detected
    if goalpost_box and ball_x is not None and ball_y is not None:
        gx, gy, gw, gh = goalpost_box

        # Define actual box edges
        left = gx - gw // 2
        right = gx + gw // 2
        top = gy - gh // 2
        bottom = gy + gh // 2

        # Draw zone boxes
        center_box = (left + gw//3, top + gh//3, right - gw//3, bottom - gh//3)
        top_left_box = (left, top, left + gw//3, top + gh//3)
        top_right_box = (right - gw//3, top, right, top + gh//3)
        bottom_left_box = (left, bottom - gh//3, left + gw//3, bottom)
        bottom_right_box = (right - gw//3, bottom - gh//3, right, bottom)

        for x1, y1, x2, y2 in [center_box, top_left_box, top_right_box, bottom_left_box, bottom_right_box]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Shot classification
        if ball_x < left or ball_x > right or ball_y < top or ball_y > bottom:
            label = "Out of Target"
        elif top <= ball_y < top + gh//3:
            label = "Top Left" if ball_x < gx else "Top Right"
        elif bottom - gh//3 <= ball_y <= bottom:
            label = "Bottom Left" if ball_x < gx else "Bottom Right"
        else:
            label = "Center"

        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return frame


def detect_player_pose(frame):
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    pose_estimator = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_estimator.process(frame_rgb)

    pose_landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            pose_landmarks.append((x, y))

        # Optional: draw pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame, pose_landmarks

def check_throwing_form(pose_landmarks, ball_x, ball_y, frame):
    if len(pose_landmarks) < 17:
        return False  # Not all required landmarks detected

    right_shoulder = pose_landmarks[12]
    right_elbow = pose_landmarks[14]
    right_wrist = pose_landmarks[16]

    # Check if elbow is higher than shoulder
    elbow_above_shoulder = right_elbow[1] < right_shoulder[1]

    # Check if ball is near wrist (hand holding the ball)
    distance_to_ball = math.hypot(ball_x - right_wrist[0], ball_y - right_wrist[1])
    ball_in_hand = distance_to_ball < 40  # you can tune this threshold

    if elbow_above_shoulder and ball_in_hand:
        cv2.putText(frame, "Good form: Elbow above shoulder", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return True
    elif ball_in_hand:
        cv2.putText(frame, "Bad form: Elbow below shoulder", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return False
    
def get_ball_position(results, ball_class_index):
    for r in results:
        for box, cls in zip(r.boxes.xywh, r.boxes.cls):
            if int(cls) == ball_class_index:
                x, y, _, _ = map(int, box.tolist())
                return x, y
    return None, None