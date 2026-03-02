import os
import random
import shutil

random.seed(42)

# =========================
# SEGMENTATION SPLIT
# =========================
seg_img = "dataset/segmentation/images"
seg_mask = "dataset/segmentation/masks"

base_seg = "dataset/segmentation"

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(base_seg, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_seg, split, "masks"), exist_ok=True)

files = os.listdir(seg_img)
random.shuffle(files)

n = len(files)
train_end = int(0.7 * n)
val_end = int(0.85 * n)

train_files = files[:train_end]
val_files = files[train_end:val_end]
test_files = files[val_end:]

def copy_seg(files, split):
    for file in files:
        shutil.copy(
            os.path.join(seg_img, file),
            os.path.join(base_seg, split, "images", file)
        )
        mask_name = os.path.splitext(file)[0] + ".png"
        shutil.copy(
            os.path.join(seg_mask, mask_name),
            os.path.join(base_seg, split, "masks", mask_name)
        )

copy_seg(train_files, "train")
copy_seg(val_files, "val")
copy_seg(test_files, "test")

print("Segmentation split done.")

# =========================
# CLASSIFICATION SPLIT
# =========================
base_cls = "dataset/classification"

for split in ["train", "val", "test"]:
    for category in ["crack", "non_crack"]:
        os.makedirs(os.path.join(base_cls, split, category), exist_ok=True)

for category in ["crack", "non_crack"]:
    
    src_folder = os.path.join(base_cls, category)
    files = os.listdir(src_folder)
    random.shuffle(files)

    n = len(files)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    train_files = files[:train_end]
    val_files = files[train_end:val_end]
    test_files = files[val_end:]

    for file in train_files:
        shutil.copy(
            os.path.join(src_folder, file),
            os.path.join(base_cls, "train", category, file)
        )

    for file in val_files:
        shutil.copy(
            os.path.join(src_folder, file),
            os.path.join(base_cls, "val", category, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(src_folder, file),
            os.path.join(base_cls, "test", category, file)
        )

print("Classification split done.")