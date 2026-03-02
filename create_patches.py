import os
import cv2
import numpy as np

# ==========================
# FOLDER PATHS
# ==========================
image_folder = "images"
mask_folder = "masks"

output_crack_img = "patches/crack/images"
output_crack_mask = "patches/crack/masks"

output_non_img = "patches/non_crack/images"
output_non_mask = "patches/non_crack/masks"

# Create folders
os.makedirs(output_crack_img, exist_ok=True)
os.makedirs(output_crack_mask, exist_ok=True)
os.makedirs(output_non_img, exist_ok=True)
os.makedirs(output_non_mask, exist_ok=True)

# ==========================
# SETTINGS
# ==========================
patch_size = 256
crack_threshold = 0.01   # 1% crack pixels

crack_count = 0
non_crack_count = 0

# ==========================
# PATCH CREATION
# ==========================
for filename in os.listdir(image_folder):

    img_path = os.path.join(image_folder, filename)
    mask_path = os.path.join(mask_folder, filename)

    if not os.path.exists(mask_path):
        continue

    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, 0)

    if image is None or mask is None:
        continue

    h, w = mask.shape

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):

            img_patch = image[y:y+patch_size, x:x+patch_size]
            mask_patch = mask[y:y+patch_size, x:x+patch_size]

            crack_pixels = np.sum(mask_patch > 0)
            crack_ratio = crack_pixels / (patch_size * patch_size)

            if crack_ratio > crack_threshold:
                cv2.imwrite(
                    f"{output_crack_img}/crack_{crack_count}.jpg",
                    img_patch
                )
                cv2.imwrite(
                    f"{output_crack_mask}/crack_{crack_count}.png",
                    mask_patch
                )
                crack_count += 1
            else:
                cv2.imwrite(
                    f"{output_non_img}/non_{non_crack_count}.jpg",
                    img_patch
                )
                cv2.imwrite(
                    f"{output_non_mask}/non_{non_crack_count}.png",
                    mask_patch
                )
                non_crack_count += 1

print("Done.")
print("Crack patches:", crack_count)
print("Non-crack patches:", non_crack_count)