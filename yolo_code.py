from ultralytics import YOLO

model = YOLO("training/runs/detect/train3/weights/best.pt")

results = model.predict('input_videos/jacob-short.mp4', save=True, show=True)

count = 0

for r in results:
    print(r.boxes.xywh)
    if r.boxes.xywh.nelement() > 0:
        count += 1

print("Count: " + str(count))