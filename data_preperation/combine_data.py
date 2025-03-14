# Imports
import os
import shutil

# Directories
train_dir = '../train'
val_dir = '../val'
# New directory to hold combined data
combined_dir = '../combined'

# Create the combined directory if it doesn't exist
os.makedirs(combined_dir, exist_ok=True)

# Get the classes from the training and validation directories
train_classes = [d for d in os.listdir(train_dir) if
                 os.path.isdir(os.path.join(train_dir, d))]
val_classes = [d for d in os.listdir(val_dir) if
               os.path.isdir(os.path.join(val_dir, d))]

# Combine the images from the training and validation directories
for class_name in train_classes:
    # Create a directory for the class in the combined directory
    class_combined_path = os.path.join(combined_dir, class_name)
    os.makedirs(class_combined_path, exist_ok=True)

    # Move images from training directory
    train_class_path = os.path.join(train_dir, class_name)
    print(train_class_path)
    for img in os.listdir(train_class_path):
        src_path = os.path.join(train_class_path, img)
        dst_path = os.path.join(class_combined_path, img)
        shutil.copy(src_path, dst_path)

    # Move images from validation directory
    val_class_path = os.path.join(val_dir, class_name)
    if os.path.exists(val_class_path):
        for img in os.listdir(val_class_path):
            src_path = os.path.join(val_class_path, img)
            dst_path = os.path.join(class_combined_path, img)
            shutil.copy(src_path, dst_path)

print("Images combined successfully into", combined_dir)
