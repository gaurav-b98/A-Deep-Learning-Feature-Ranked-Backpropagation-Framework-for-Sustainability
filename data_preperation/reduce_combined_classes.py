# Imports
import os
import random
import shutil

# Path to the source directory with all classes
data_dir = '../combined'

# Path to the destination directory with the selected classes
new_data_dir = '../combined_200'

# Number of classes to select
num_classes = 200

# Get all classes
all_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Select random classes
selected_classes = random.sample(all_classes, num_classes)

# Function to copy selected classes
def copy_selected_classes(src_dir, dest_dir, selected_classes):
    """
    Copy selected classes from the source directory to the destination directory.
    Args:
        src_dir: Source directory with all classes
        dest_dir: Destination directory to copy the selected classes
        selected_classes: List of selected classes
    """
    for class_name in selected_classes:
        src_class_path = os.path.join(src_dir, class_name)
        dest_class_path = os.path.join(dest_dir, class_name)
        shutil.copytree(src_class_path, dest_class_path)
        print(f"Copied {class_name} from {src_dir} to {dest_dir}")


# Create the destination directory
os.makedirs(new_data_dir, exist_ok=True)

# Copy selected classes
copy_selected_classes(data_dir, new_data_dir, selected_classes)

print(f"Dataset reduction complete with the {num_classes} classes.")
