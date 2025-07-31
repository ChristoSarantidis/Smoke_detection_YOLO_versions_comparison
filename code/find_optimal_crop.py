# Script to determine best crop percentage using detection-aware strategy.
# It expands each detection, crops around it, and evaluates average confidence for each crop scale.

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
import matplotlib.ticker as mticker
import csv

# === Parameters ===
val_images_dir = ".../fold6/test/images" # directory containing images
crop_scales = [ 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
image_size = 640
padding_ratio = 0.2  # Expand detection bounding box by 20%

# === Collect all image paths ===
image_paths = glob(os.path.join(val_images_dir, "*.jpg"))

###############################################################################################yolov13
# === Load model ===
model_path = ".../weights/best.pt"
model = YOLO(model_path)

log_rows = []
model_name = "YOLOv13nano"



average_confidences = []

for scale in crop_scales:
    confidences = []

    for img_path in tqdm(image_paths, desc=f"Processing scale {scale}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run inference on full image to get detections
        results = model(img, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            continue  # Skip if no detections

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Determine crop size as scale * min(h, w), centered around detection
            target_crop_size = int(scale * min(h, w))

            # Center coordinates of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get crop boundaries around detection center
            half_crop = target_crop_size // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w)
            crop_y2 = min(cy + half_crop, h)

            # Fix crop size if it hits image boundaries
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue  # skip overly small crops

            resized_crop = cv2.resize(crop, (image_size, image_size))

            # Run inference on crop
            crop_result = model(resized_crop, verbose=False)
            crop_detections = crop_result[0].boxes

            for cdet in crop_detections:
                confidences.append(float(cdet.conf))

    avg_conf = np.mean(confidences) if confidences else 0.0
    average_confidences.append(avg_conf)
    log_rows.append([model_name, scale * 100, avg_conf])

with open("crop_confidences_log_v13.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "crop_percentage", "confidence"])  # header
    writer.writerows(log_rows)



# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot([int(s * 100) for s in crop_scales], average_confidences, marker='o')
plt.xlabel("Crop size (% of original image)")
plt.ylabel("Average confidence")
plt.title("Detection‑Aware Average Confidence vs Crop Percentage (YOLOv13nano)")
plt.grid(True)

# --- NEW: force ticks every 5 % from 25 to 100 ---
ax = plt.gca()                                   # current axes
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))   # step = 5
ax.set_xlim(25, 100)                              # ensure full range
ax.set_xticklabels([f"{t:.0f}%" for t in ax.get_xticks()])   # add % sign (optional)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))  # y-tick every 0.1
ax.set_ylim(0.3, 0.7)  # y-axis from 0 to 1

plt.tight_layout()
plt.savefig("YOLOv13nano_detection_aware_confidence_vs_crop.png")
plt.show()


# === Report best crop percentage ===
max_index = np.argmax(average_confidences)
best_crop_percentage = crop_scales[max_index] * 100
print(f"\n✅ Best crop percentage: {best_crop_percentage:.1f}% with average confidence: {average_confidences[max_index]:.4f}")

###############################################################################################yolov12
# === Load model ===
model_path = ".../weights/best.pt"
model = YOLO(model_path)
log_rows = []
model_name = "YOLOv12nano"

average_confidences = []

for scale in crop_scales:
    confidences = []

    for img_path in tqdm(image_paths, desc=f"Processing scale {scale}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run inference on full image to get detections
        results = model(img, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            continue  # Skip if no detections

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Determine crop size as scale * min(h, w), centered around detection
            target_crop_size = int(scale * min(h, w))

            # Center coordinates of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get crop boundaries around detection center
            half_crop = target_crop_size // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w)
            crop_y2 = min(cy + half_crop, h)

            # Fix crop size if it hits image boundaries
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue  # skip overly small crops

            resized_crop = cv2.resize(crop, (image_size, image_size))

            # Run inference on crop
            crop_result = model(resized_crop, verbose=False)
            crop_detections = crop_result[0].boxes

            for cdet in crop_detections:
                confidences.append(float(cdet.conf))

    avg_conf = np.mean(confidences) if confidences else 0.0
    average_confidences.append(avg_conf)
    log_rows.append([model_name, scale * 100, avg_conf])
    
with open("crop_confidences_log_v12.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "crop_percentage", "confidence"])  # header
    writer.writerows(log_rows)



# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot([int(s * 100) for s in crop_scales], average_confidences, marker='o')
plt.xlabel("Crop size (% of original image)")
plt.ylabel("Average confidence")
plt.title("Detection‑Aware Average Confidence vs Crop Percentage (YOLOv12nano)")
plt.grid(True)

# --- NEW: force ticks every 5 % from 25 to 100 ---
ax = plt.gca()                                   # current axes
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))   # step = 5
ax.set_xlim(25, 100)                              # ensure full range
ax.set_xticklabels([f"{t:.0f}%" for t in ax.get_xticks()])   # add % sign (optional)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))  # y-tick every 0.1
ax.set_ylim(0.3, 0.7)  # y-axis from 0 to 1

plt.tight_layout()
plt.savefig("YOLOv12nano_detection_aware_confidence_vs_crop.png")
plt.show()


# === Report best crop percentage ===
max_index = np.argmax(average_confidences)
best_crop_percentage = crop_scales[max_index] * 100
print(f"\n✅ Best crop percentage: {best_crop_percentage:.1f}% with average confidence: {average_confidences[max_index]:.4f}")


###############################################################################################yolov11
# === Load model ===
model_path = ".../weights/best.pt"
model = YOLO(model_path)

log_rows = []
model_name = "YOLOv11nano"

average_confidences = []

for scale in crop_scales:
    confidences = []

    for img_path in tqdm(image_paths, desc=f"Processing scale {scale}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run inference on full image to get detections
        results = model(img, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            continue  # Skip if no detections

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Determine crop size as scale * min(h, w), centered around detection
            target_crop_size = int(scale * min(h, w))

            # Center coordinates of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get crop boundaries around detection center
            half_crop = target_crop_size // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w)
            crop_y2 = min(cy + half_crop, h)

            # Fix crop size if it hits image boundaries
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue  # skip overly small crops

            resized_crop = cv2.resize(crop, (image_size, image_size))

            # Run inference on crop
            crop_result = model(resized_crop, verbose=False)
            crop_detections = crop_result[0].boxes

            for cdet in crop_detections:
                confidences.append(float(cdet.conf))

    avg_conf = np.mean(confidences) if confidences else 0.0
    average_confidences.append(avg_conf)
    log_rows.append([model_name, scale * 100, avg_conf])
    
    
with open("crop_confidences_log_v11.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "crop_percentage", "confidence"])  # header
    writer.writerows(log_rows)


# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot([int(s * 100) for s in crop_scales], average_confidences, marker='o')
plt.xlabel("Crop size (% of original image)")
plt.ylabel("Average confidence")
plt.title("Detection‑Aware Average Confidence vs Crop Percentage (YOLOv11nano)")
plt.grid(True)

# --- NEW: force ticks every 5 % from 25 to 100 ---
ax = plt.gca()                                   # current axes
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))   # step = 5
ax.set_xlim(25, 100)                              # ensure full range
ax.set_xticklabels([f"{t:.0f}%" for t in ax.get_xticks()])   # add % sign (optional)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))  # y-tick every 0.1
ax.set_ylim(0.3, 0.7)  # y-axis from 0 to 1

plt.tight_layout()
plt.savefig("YOLOv11nano_detection_aware_confidence_vs_crop.png")
plt.show()


# === Report best crop percentage ===
max_index = np.argmax(average_confidences)
best_crop_percentage = crop_scales[max_index] * 100
print(f"\n✅ Best crop percentage: {best_crop_percentage:.1f}% with average confidence: {average_confidences[max_index]:.4f}")


###############################################################################################yolov10
# === Load model ===
model_path = ".../weights/best.pt"
model = YOLO(model_path)

log_rows = []
model_name = "YOLOv10nano" 

average_confidences = []

for scale in crop_scales:
    confidences = []

    for img_path in tqdm(image_paths, desc=f"Processing scale {scale}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run inference on full image to get detections
        results = model(img, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            continue  # Skip if no detections

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Determine crop size as scale * min(h, w), centered around detection
            target_crop_size = int(scale * min(h, w))

            # Center coordinates of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get crop boundaries around detection center
            half_crop = target_crop_size // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w)
            crop_y2 = min(cy + half_crop, h)

            # Fix crop size if it hits image boundaries
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue  # skip overly small crops

            resized_crop = cv2.resize(crop, (image_size, image_size))

            # Run inference on crop
            crop_result = model(resized_crop, verbose=False)
            crop_detections = crop_result[0].boxes

            for cdet in crop_detections:
                confidences.append(float(cdet.conf))

    avg_conf = np.mean(confidences) if confidences else 0.0
    average_confidences.append(avg_conf)
    log_rows.append([model_name, scale * 100, avg_conf])


with open("crop_confidences_log_v10.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "crop_percentage", "confidence"])  # header
    writer.writerows(log_rows)



# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot([int(s * 100) for s in crop_scales], average_confidences, marker='o')
plt.xlabel("Crop size (% of original image)")
plt.ylabel("Average confidence")
plt.title("Detection‑Aware Average Confidence vs Crop Percentage (YOLOv10nano)")
plt.grid(True)

# --- NEW: force ticks every 5 % from 25 to 100 ---
ax = plt.gca()                                   # current axes
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))   # step = 5
ax.set_xlim(25, 100)                              # ensure full range
ax.set_xticklabels([f"{t:.0f}%" for t in ax.get_xticks()])   # add % sign (optional)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))  # y-tick every 0.1
ax.set_ylim(0.3, 0.7)  # y-axis from 0 to 1

plt.tight_layout()
plt.savefig("YOLOv10nano_detection_aware_confidence_vs_crop.png")
plt.show()


# === Report best crop percentage ===
max_index = np.argmax(average_confidences)
best_crop_percentage = crop_scales[max_index] * 100
print(f"\n✅ Best crop percentage: {best_crop_percentage:.1f}% with average confidence: {average_confidences[max_index]:.4f}")


###############################################################################################yolov9tiny
# === Load model ===
model_path = ".../weights/best.pt"
model = YOLO(model_path)

log_rows = []
model_name = "YOLOv9tiny" 

average_confidences = []

for scale in crop_scales:
    confidences = []

    for img_path in tqdm(image_paths, desc=f"Processing scale {scale}"):
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Run inference on full image to get detections
        results = model(img, verbose=False)
        detections = results[0].boxes

        if len(detections) == 0:
            continue  # Skip if no detections

        for det in detections:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            box_w, box_h = x2 - x1, y2 - y1

            # Determine crop size as scale * min(h, w), centered around detection
            target_crop_size = int(scale * min(h, w))

            # Center coordinates of detection
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Get crop boundaries around detection center
            half_crop = target_crop_size // 2
            crop_x1 = max(cx - half_crop, 0)
            crop_y1 = max(cy - half_crop, 0)
            crop_x2 = min(cx + half_crop, w)
            crop_y2 = min(cy + half_crop, h)

            # Fix crop size if it hits image boundaries
            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.shape[0] < 5 or crop.shape[1] < 5:
                continue  # skip overly small crops

            resized_crop = cv2.resize(crop, (image_size, image_size))

            # Run inference on crop
            crop_result = model(resized_crop, verbose=False)
            crop_detections = crop_result[0].boxes

            for cdet in crop_detections:
                confidences.append(float(cdet.conf))

    avg_conf = np.mean(confidences) if confidences else 0.0
    average_confidences.append(avg_conf)
    log_rows.append([model_name, scale * 100, avg_conf])


with open("crop_confidences_log_v9.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["model", "crop_percentage", "confidence"])  # header
    writer.writerows(log_rows)


# === Plot results ===
plt.figure(figsize=(10, 6))
plt.plot([int(s * 100) for s in crop_scales], average_confidences, marker='o')
plt.xlabel("Crop size (% of original image)")
plt.ylabel("Average confidence")
plt.title("Detection‑Aware Average Confidence vs Crop Percentage (YOLOv9tiny)")
plt.grid(True)

# --- NEW: force ticks every 5 % from 25 to 100 ---
ax = plt.gca()                                   # current axes
ax.xaxis.set_major_locator(mticker.MultipleLocator(5))   # step = 5
ax.set_xlim(25, 100)                              # ensure full range
ax.set_xticklabels([f"{t:.0f}%" for t in ax.get_xticks()])   # add % sign (optional)

ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))  # y-tick every 0.1
ax.set_ylim(0.3, 0.7)  # y-axis from 0 to 1

plt.tight_layout()
plt.savefig("YOLOv9tiny_detection_aware_confidence_vs_crop.png")
plt.show()


# === Report best crop percentage ===
max_index = np.argmax(average_confidences)
best_crop_percentage = crop_scales[max_index] * 100
print(f"\n✅ Best crop percentage: {best_crop_percentage:.1f}% with average confidence: {average_confidences[max_index]:.4f}")
