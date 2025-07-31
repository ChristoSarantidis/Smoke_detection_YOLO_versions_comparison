import os
import time
import pandas as pd
from ultralytics import YOLO


folds = ['fold1.yaml', 'fold2.yaml', 'fold3.yaml', 'fold4.yaml', 'fold5.yaml']

results_dir = 'results'
os.makedirs(results_dir, exist_ok=True)

#####################################################################################################yolov13n

from ultralytics.nn.modules.block import DSC3k2
model_path = 'yolov13n.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=100, imgsz=640, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)

#####################################################################################################yolov12n
model_path = 'yolov12n.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=100, imgsz=640, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)


#####################################################################################################yolov11n

model_path = 'yolov11n.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=100, imgsz=640, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)
#####################################################################################################yolov10n
model_path = 'yolov10n.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=100, imgsz=640, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)
#####################################################################################################yolov9t
model_path = 'yolov9t.pt'  # Change to yolov10n.pt, etc.


# Store results
fold_results = []

for fold_yaml in folds:
    fold_name = os.path.splitext(os.path.basename(fold_yaml))[0]
    print(f"Training on {fold_name}...")

    # Load model
    model = YOLO(model_path)

    # --- Training ---
    start_train = time.time()
    model.train(data=fold_yaml, epochs=100, imgsz=640, project='runs_crossval', name=fold_name, device=[0,1,2], batch=39, save_txt=True)
    train_time = time.time() - start_train

    # --- Validation ---
    start_val = time.time()
    metrics = model.val(data=fold_yaml)
    val_time = time.time() - start_val

    # Extract metrics
    result = {
        'model': os.path.basename(model_path),
        'fold': fold_name,
        'precision': round(metrics.box.mp, 3),         # ✅ no parentheses
        'recall': round(metrics.box.mr, 3),            # ✅
        'f1': round(sum(metrics.box.f1)/len(metrics.box.f1), 3),  # still valid
        'map50': round(metrics.box.map50, 3),          # ✅
        'train_time_sec': round(train_time, 2),
        'val_time_sec': round(val_time, 2)
    }

    fold_results.append(result)

# Save results to CSV
df = pd.DataFrame(fold_results)
csv_filename = os.path.join(results_dir, f'{os.path.splitext(os.path.basename(model_path))[0]}_crossval_results.csv')
df.to_csv(csv_filename, index=False)

print(f"\n✅ Results saved to: {csv_filename}")
print(df)