# Imports
import os
import shutil

# Path to the folder containing all the images
source_folder = "../train"

# Path to the folder where images will be sorted by class
destination_folder = "../train"

# Iterate over all the files in the source folder
for filename in os.listdir(source_folder):
    # Check if the file is an image
    if filename.endswith(".JPEG") or filename.endswith(".jpg") or filename.endswith(".png"):
        print(filename)
        # Extract the class name and image number from the filename
        # Example filename: n01443537_1_n01443537.JPEG
        parts = filename.split('_')
        class_name = parts[0]
        image_number = parts[1]

        # Create a folder for the class if it doesn't exist
        class_folder = os.path.join(destination_folder, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Rename the image to the image number
        new_filename = f"{image_number}.JPEG"

        # Move the image to the class folder
        source_path = os.path.join(source_folder, filename)
        destination_path = os.path.join(class_folder, new_filename)
        shutil.move(source_path, destination_path)

print("Images have been renamed and moved to class-based folders.")
