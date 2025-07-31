import os

def rename_images_sequentially(folder_path):
    """
    Rename all images in the specified folder to sequential names: odysseas1, odysseas2, ..., odysseasN.
    If images named odysseas... exist, they are also renamed to ensure sequential numbering.
    :param folder_path: Path to the folder containing images.
    """
    try:
        # Check if the folder exists
        if not os.path.exists(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return

        # Get a list of all image files in the folder
        image_extensions = ('.txt')
        images = [file for file in os.listdir(folder_path) if file.lower().endswith(image_extensions)]

        if not images:
            print("No image files found in the specified folder.")
            return

        # Sort the images to avoid conflicts while renaming
        images.sort()

        # Rename images sequentially
        for i, image in enumerate(images, start=1):
            old_path = os.path.join(folder_path, image)
            new_name = f"fire_cam_{i}{os.path.splitext(image)[1]}"  # Preserve original file extension
            new_path = os.path.join(folder_path, new_name)

            # Rename the image
            os.rename(old_path, new_path)
            print(f"Renamed: {image} -> {new_name}")

        print(f"All {len(images)} images have been renamed successfully.")
        print(f"The total number of images is {len(images)}, and the last image is named 'fire_sat_{len(images)}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
folder_path = input("Enter the folder path containing the images: ").strip()
rename_images_sequentially(folder_path)
