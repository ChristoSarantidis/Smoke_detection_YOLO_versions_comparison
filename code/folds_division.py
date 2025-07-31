import os
import shutil
from sklearn.model_selection import KFold, train_test_split

#  CHANGE THESE PATHS 
images_folder = 'images'  # Directory containing grayscale images
labels_folder = 'labels'
output_base_folder = 'crossval_folds'

#  Create list of all image filenames (without extension) 
image_ids = [f.replace('.jpg', '') for f in os.listdir(images_folder) if f.endswith('.jpg')]
image_ids.sort()  # Make sure the order is consistent

#  5-Fold Split ####################n-splits
kf = KFold(n_splits=5, shuffle=True, random_state=4)

for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(image_ids), start=1):
    print(f"Processing fold{fold_idx}...")
    
    # Split into trainval and test
    trainval_ids = [image_ids[i] for i in trainval_idx]
    test_ids = [image_ids[i] for i in test_idx]
    
    # Further split trainval into train and val (90% train / 10% val)
    train_ids, val_ids = train_test_split(trainval_ids, test_size=0.1, random_state=fold_idx)
    
    # Create fold folder structure
    fold_path = os.path.join(output_base_folder, f'fold{fold_idx}')
    for subset in ['train', 'val', 'test']:
        os.makedirs(os.path.join(fold_path, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(fold_path, subset, 'labels'), exist_ok=True)
    
    # Copy files
    def copy_files(ids_list, subset):
        for img_id in ids_list:
            # Image and label filenames
            img_src = os.path.join(images_folder, f"{img_id}.jpg")
            lbl_src = os.path.join(labels_folder, f"{img_id}.txt")
            
            img_dst = os.path.join(fold_path, subset, 'images', f"{img_id}.jpg")
            lbl_dst = os.path.join(fold_path, subset, 'labels', f"{img_id}.txt")
            
            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)

    copy_files(train_ids, 'train')
    copy_files(val_ids, 'val')
    copy_files(test_ids, 'test')

    print(f"✔ Fold {fold_idx} done. Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}\n")

print("✅ All folds created successfully!")




#data.yaml creation
'''
import os
import yaml  # Make sure PyYAML is installed

# Number of classes and class names
nc = 1
class_names = ["smoke"]

output_base_folder = 'crossval_folds_from_excel'  # Same as before

for fold_num in range(1, 6):
    fold_path = os.path.join(output_base_folder, f'fold{fold_num}')
    yaml_path = os.path.join(fold_path, 'data.yaml')

    data_dict = {
        'train': os.path.join(fold_path, 'train/images'),
        'val': os.path.join(fold_path, 'val/images'),
        'nc': nc,
        'names': class_names
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data_dict, f)

    print(f"✅ data.yaml created for fold {fold_num}: {yaml_path}")

'''