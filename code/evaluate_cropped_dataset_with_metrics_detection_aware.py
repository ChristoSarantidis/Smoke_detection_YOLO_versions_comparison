import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from ultralytics import YOLO
import yaml
import time

# === CONFIGURATION ===
BASE_DIR = "dir"  # directory where trained model folders are
DATASET_DIR = "dir" # dataset heq or normal

CROPPED_DATASET_DIR = "....cropped_eval_dataset" # output directory for cropped images
IMAGE_SIZE = 640
OUTPUT_CSV = "cropped_eval_metrics.csv"

# Crop percentage by algorithm
CROP_PERCENTAGE_BY_MODEL = {
    "yolov9": 1,
    "yolov10": 0.95,
    "yolov11": 0.95,
    "yolov12": 1,
    "yolov13": 1
}

# Choose algorithm to run
target_algorithm = "yolov12"
crop_percent = CROP_PERCENTAGE_BY_MODEL[target_algorithm]
os.makedirs(CROPPED_DATASET_DIR, exist_ok=True)

model_folders = sorted([
    f for f in os.listdir(BASE_DIR)
    if f.startswith(target_algorithm) and "fold" in f
])

results = []

for model_folder in model_folders:
    print(f"🔍 Evaluating {model_folder}")
    fold_num = model_folder.split("fold")[-1]
    model_path = os.path.join(BASE_DIR, model_folder, "weights", "best.pt")
    model = YOLO(model_path)

    yaml_file_path = os.path.join(DATASET_DIR, f"fold{fold_num}.yaml")
    with open(yaml_file_path, 'r') as yfile:
        fold_yaml = yaml.safe_load(yfile)

    val_img_dir = os.path.join(DATASET_DIR, f"fold{fold_num}", "test", "images")
    val_lbl_dir = os.path.join(DATASET_DIR, f"fold{fold_num}", "test", "labels")

    crop_fold_dir = os.path.join(CROPPED_DATASET_DIR, f"{target_algorithm}_fold{fold_num}")
    crop_images_dir = os.path.join(crop_fold_dir, "images")
    crop_labels_dir = os.path.join(crop_fold_dir, "labels")
    os.makedirs(crop_images_dir, exist_ok=True)
    os.makedirs(crop_labels_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(val_img_dir, "*.jpg")))
    for img_path in tqdm(image_paths):
        filename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(val_lbl_dir, f"{filename}.txt")

        img = cv2.imread(img_path)
        if img is None or not os.path.exists(label_path):
            continue
        h_img, w_img = img.shape[:2]

        with open(label_path, 'r') as f:
            lines = f.readlines()
        gt_boxes = []
        for line in lines:
            cls, x, y, w, h = map(float, line.strip().split())
            x_abs = x * w_img
            y_abs = y * h_img
            w_abs = w * w_img
            h_abs = h * h_img
            gt_boxes.append((cls, x_abs, y_abs, w_abs, h_abs))

        results_det = model(img, verbose=False)
        detections = results_det[0].boxes
        if len(detections) == 0:
            continue

        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            crop_sz = int(crop_percent * min(h_img, w_img))
            half_crop = crop_sz // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w_img)
            crop_y2 = min(cy + half_crop, h_img)

            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop_img.shape[0] < 10 or crop_img.shape[1] < 10:
                continue
            resized_img = cv2.resize(crop_img, (IMAGE_SIZE, IMAGE_SIZE))

            out_img_path = os.path.join(crop_images_dir, f"{filename}_{idx}.jpg")
            cv2.imwrite(out_img_path, resized_img)

            crop_labels = []
            for cls, x_abs, y_abs, w_abs, h_abs in gt_boxes:
                if crop_x1 < x_abs < crop_x2 and crop_y1 < y_abs < crop_y2:
                    new_x = (x_abs - crop_x1) / (crop_x2 - crop_x1)
                    new_y = (y_abs - crop_y1) / (crop_y2 - crop_y1)
                    new_w = w_abs / (crop_x2 - crop_x1)
                    new_h = h_abs / (crop_y2 - crop_y1)

                    if (
                        0.0 <= new_x <= 1.0 and
                        0.0 <= new_y <= 1.0 and
                        0.01 <= new_w <= 1.0 and
                        0.01 <= new_h <= 1.0
                    ):
                        crop_labels.append(f"{int(cls)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

            with open(os.path.join(crop_labels_dir, f"{filename}_{idx}.txt"), 'w') as f:
                f.write("\n".join(crop_labels))

    cropped_yaml_path = os.path.join(crop_fold_dir, "cropped_data.yaml")
    with open(cropped_yaml_path, 'w') as f:
        f.write(f"path: {crop_fold_dir}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write(f"nc: {fold_yaml['nc']}\n")
        f.write(f"names: {fold_yaml['names']}\n")

    start_val = time.time()
    metrics = model.val(data=cropped_yaml_path, verbose=False)
    val_time_sec = round(time.time() - start_val, 2)

    results.append({
        "algorithm": target_algorithm,
        "fold": fold_num,
        "crop_percentage": crop_percent,
        "precision": round(metrics.box.mp, 3),
        "recall": round(metrics.box.mr, 3),
        "f1": round(sum(metrics.box.f1)/len(metrics.box.f1), 3) if metrics.box.f1 else 0.0,
        "map50": round(metrics.box.map50, 3),
        "map50_95": round(metrics.box.map, 3),
        "val_time_sec": val_time_sec,
        "images_used": len(image_paths)
    })

df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Final metrics saved to {OUTPUT_CSV}")
print(df)

        