import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

num_classes = 3

# Load model and set to eval mode
model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
model.load_state_dict(torch.load("training-rcnn\\faster_rcnn_handball.pth", weights_only=True))  # your trained weights
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Open video
cap = cv2.VideoCapture("input_videos/jacob-short.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_videos/output_frcnn.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Label map (match your dataset!)
class_names = {0: "__background__", 1: "post", 2: "handball"}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to tensor and normalize
    input_tensor = F.to_tensor(frame).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)[0]

    # Init
    goalpost_box = None
    ball_x, ball_y = None, None

    # Parse predictions
    for box, label, score in zip(outputs['boxes'], outputs['labels'], outputs['scores']):
        if score < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if class_names[label.item()] == "post":
            goalpost_box = (x1, y1, x2, y2)

            # Define sections
            w, h = x2 - x1, y2 - y1
            gx, gy = (x1 + x2) // 2, (y1 + y2) // 2

            center_box = (gx - w // 4, gy, gx + w // 4, gy + h // 3)
            top_left_box = (x1, y1, x1 + w // 4, y1 + h // 3)
            top_right_box = (x2 - w // 4, y1, x2, y1 + h // 3)
            bottom_left_box = (x1, y2 - h // 3, x1 + w // 4, y2)
            bottom_right_box = (x2 - w // 4, y2 - h // 3, x2, y2)

            for box in [center_box, top_left_box, top_right_box, bottom_left_box, bottom_right_box]:
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        elif class_names[label.item()] == "handball":
            ball_x, ball_y = cx, cy
            cv2.circle(frame, (ball_x, ball_y), 6, (0, 255, 0), -1)

    # Classify shot
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

        cv2.putText(frame, position_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Faster R-CNN Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
