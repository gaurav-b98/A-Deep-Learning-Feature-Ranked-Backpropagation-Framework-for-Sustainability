# Imports
import os
import shutil

# Path to the folder containing the images to be sorted
source_folder = "/home/ubuntu/dataset/val"

# Path to the folder where images will be sorted by class
destination_folder = "/home/ubuntu/dataset/val"

# Iterate over each file in the source folder
for filename in os.listdir(source_folder):
    print(filename)
    # Check if the file is an image
    if filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename)
        # Extract the image number and class name from the filename
        # Example filename: ILSVRC2012_2571_n01774384.JPEG
        parts = filename.split('_')
        image_number = parts[2]
        class_name = parts[-1].split('.')[0]

        # Create a folder for the class if it doesn't exist
        class_folder = os.path.join(destination_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
        # Rename the image to the image number
        file_extension = filename.split('.')[-1]
        new_filename = f"{image_number}.{file_extension}"

        # Move the image to the class folder
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(class_folder, new_filename)
        shutil.move(source_path, destination_path)

print("Validation images have been renamed and moved to class-based folders.")
