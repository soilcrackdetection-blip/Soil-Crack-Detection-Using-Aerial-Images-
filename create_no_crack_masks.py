import os
import cv2
import numpy as np

# Input and output folders
image_folder = "no crack images"
mask_folder = "no crack masks"

# Create mask folder if not exists
os.makedirs(mask_folder, exist_ok=True)

for filename in os.listdir(image_folder):

    image_path = os.path.join(image_folder, filename)

    image = cv2.imread(image_path)

    if image is None:
        continue

    h, w = image.shape[:2]

    # Create full black mask
    black_mask = np.zeros((h, w), dtype=np.uint8)

    # Save mask with same filename
    mask_path = os.path.join(mask_folder, filename)

    cv2.imwrite(mask_path, black_mask)

    print(f"Mask created for {filename}")

print("All non-crack masks created successfully.")