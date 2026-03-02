import os
import cv2

base_path = "dataset/segmentation"
splits = ["train", "val", "test"]

for split in splits:
    img_folder = os.path.join(base_path, split, "images")
    mask_folder = os.path.join(base_path, split, "masks")
    
    for filename in os.listdir(img_folder):
        img_path = os.path.join(img_folder, filename)
        mask_path = os.path.join(mask_folder, filename.replace(".jpg", ".png"))
        
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        
        if image is None or mask is None:
            continue
        
        resized_img = cv2.resize(image, (256, 256))
        resized_mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(img_path, resized_img)
        cv2.imwrite(mask_path, resized_mask)

print("Segmentation resizing completed.")