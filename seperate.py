import os
import shutil

# Path to the folder containing both images and XML files
source_dir = "C:\\Users\\Jacob\\Desktop\\Thesis\\Code\\training-rcnn\\handball-detection-8\\valid"

# Paths to the destination folders
images_dir = os.path.join(source_dir, "images")
annotations_dir = os.path.join(source_dir, "annotations")

# Create the destination folders if they don't exist
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Loop through the files and move them
for filename in os.listdir(source_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        shutil.move(os.path.join(source_dir, filename), os.path.join(images_dir, filename))
    elif filename.endswith(".xml"):
        shutil.move(os.path.join(source_dir, filename), os.path.join(annotations_dir, filename))

print("Files separated successfully.")
