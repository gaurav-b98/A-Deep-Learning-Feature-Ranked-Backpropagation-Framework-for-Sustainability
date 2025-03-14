# Imports
import os
import shutil
import random

# Set random seed
random.seed(42)

# Define source and destination directories
source_dir = '../combined'
train_dir = '../train_1000'
test_dir = '../test_1000'
val_dir = '../val_1000'

# Define split ratios
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Create destination directories
for dir_path in [train_dir, test_dir, val_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Loop through each class folder in the source directory
for class_folder in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_folder)
    if not os.path.isdir(class_path):
        continue

    # Get list of images in the class folder
    images = [img for img in os.listdir(class_path) if img.endswith('.JPEG')]
    random.shuffle(images)

    # Calculate the number of images for each split
    train_count = int(len(images) * train_ratio)
    val_count = int(len(images) * val_ratio)

    # Split the images into train, test, and validation sets
    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    # Function to move images to destination directory
    def move_images(image_list, dest_dir):
        """
        Move images to the destination directory
        Args:
            image_list: list of image filenames
            dest_dir: destination directory
        """
        # Create a subdirectory for the class in the destination directory
        class_dest_dir = os.path.join(dest_dir, class_folder)
        os.makedirs(class_dest_dir, exist_ok=True)
        for img in image_list:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(class_dest_dir, img))

    # Move images to the destination directories
    move_images(train_images, train_dir)
    move_images(val_images, val_dir)
    move_images(test_images, test_dir)

print("Dataset split completed.")
