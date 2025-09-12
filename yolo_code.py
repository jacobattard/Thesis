from yolo_methods import (
    load_model,
    setup_video_io,
    process_frame,
    detect_player_pose,
    check_throwing_form,
    get_ball_position,
)
import cv2
from datetime import datetime

model_path = "training-rcnn/runs/detect/train11/weights/best.pt"
video_path = "input_videos/jacob.mp4"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"output_videos/output_{timestamp}.mp4"

# Load YOLO model and indices for goalpost and ball
model, post_idx, ball_idx = load_model(model_path)

# Set up video reader and writer
cap, out = setup_video_io(video_path, output_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame)

    # Get ball position
    ball_x, ball_y = get_ball_position(results, ball_idx)

    # Detect player pose
    frame, player_pose = detect_player_pose(frame)

    # Analyze throwing form
    if ball_x is not None and ball_y is not None and player_pose is not None:
        check_throwing_form(player_pose, ball_x, ball_y, frame)

    # Draw goal zones & annotate
    frame = process_frame(frame, results, post_idx, ball_idx)

    # Show and save
    cv2.imshow("Handball Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
