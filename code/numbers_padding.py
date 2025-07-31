import os
import re

# Folder containing the annotation files
annotation_folder = 'C:/Users/ionio/Desktop/fire_vol2.2/smoke_camera/all/images'
output_file = 'added_numbers.txt'

renamed_files = []

# Pattern to match files like fire_sat_2.txt, fire_sat_10.txt, etc.
pattern = re.compile(r'^fire_cam_(\d+)\.jpg$')

for filename in os.listdir(annotation_folder):
    match = pattern.match(filename)
    if match:
        number = int(match.group(1))
        new_number = f'{number:03d}'  # Pads to 3 digits
        new_name = f'fire_cam_{new_number}.jpg'
        old_path = os.path.join(annotation_folder, filename)
        new_path = os.path.join(annotation_folder, new_name)

        os.rename(old_path, new_path)
        renamed_files.append(new_name)
        print(f"Renamed: {filename} → {new_name}")

# Save the renamed filenames to added_numbers.txt
with open(os.path.join(annotation_folder, output_file), 'w') as f:
    for name in renamed_files:
        f.write(name + '\n')

print(f"\nDone! {len(renamed_files)} files renamed. Saved to '{output_file}'.")
