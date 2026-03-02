import os
import cv2

base_path = "dataset/classification"

splits = ["train", "val", "test"]
classes = ["crack", "non_crack"]

for split in splits:
    for cls in classes:
        folder = os.path.join(base_path, split, cls)
        
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            resized = cv2.resize(image, (224, 224))
            
            cv2.imwrite(img_path, resized)

print("Classification resizing completed.")