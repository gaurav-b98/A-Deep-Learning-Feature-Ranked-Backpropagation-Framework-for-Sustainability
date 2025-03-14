# System Imports
import os
# Third-party Imports
import csv
import yaml
import numpy as np
import random
# PyTorch Imports
import torch
# Custom Imports
from utils.logging import setup_logger, log_event
from utils.data_loader import get_data_loaders
# Import models
from models.resnet import ResNetStandard, ResNetSelective
from models.vggnet import VGGNetStandard, VGGNetSelective
from models.alexnet import AlexNetStandard, AlexNetSelective
# Import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
# Import the class mapping
from classes import IMAGENET2012_CLASSES

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def test_model(config_path):
    """
    Test the model using the test data loader and log the results
    Args:
        config_path (str): Path to the configuration file
    """
    # Load the configuration file
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Extract configuration parameters
    log_dir = config['logging']['log_dir']
    model_type = config['model']['model_type']
    architecture = config['model']['architecture']
    num_classes = config['model']['num_classes']

    if architecture == 'standard':
        all_log_dir = os.path.join(
            log_dir, f"{model_type}_{architecture}")
        model_dir = os.path.join(
            config['training']['model_save_path'],
            f"{model_type}_{architecture}")
    elif architecture == 'selective':
        schedule = config['training']['percentage_schedule']
        all_log_dir = os.path.join(
            log_dir, f"{model_type}_{architecture}_{schedule}")
        model_dir = os.path.join(
            config['training']['model_save_path'],
            f"{model_type}_{architecture}_{schedule}")

    # Setup logger
    setup_logger(all_log_dir, "test.log")
    log_event(
        f"Testing model: {model_type}, Architecture: {architecture}")

    # Model selection
    if model_type == 'alexnet':
        if architecture == 'standard':
            model = AlexNetStandard(num_classes=num_classes)
        elif architecture == 'selective':
            model = AlexNetSelective(num_classes=num_classes)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
    elif model_type == 'resnet':
        if architecture == 'standard':
            model = ResNetStandard(num_classes=num_classes)
        elif architecture == 'selective':
            model = ResNetSelective(num_classes=num_classes)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
    elif model_type == 'vggnet':
        if architecture == 'standard':
            model = VGGNetStandard(num_classes=num_classes)
        elif architecture == 'selective':
            model = VGGNetSelective(num_classes=num_classes)
        else:
            raise ValueError(f"Invalid architecture: {architecture}")
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Load the model to the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the best model
    model_save_path = os.path.join(model_dir,
                                   "model_best.pth")
    state_dict = torch.load(model_save_path, weights_only=True)
    model.load_state_dict(state_dict)

    # Get test data loader
    _, _, test_loader = get_data_loaders(config)

    # Create class mappings for the encrypted class names
    encrypted_class_names = list(IMAGENET2012_CLASSES.keys())
    index_to_encrypted = {idx: enc_name for idx,
                          enc_name in enumerate(encrypted_class_names)}
    encrypted_to_original = IMAGENET2012_CLASSES  # Already provided

    def get_class_names(label_index):
        """
        Get the encrypted and original class names from the label index
        Args:
            label_index (int): Index of the class label
        Returns:
            encrypted_name (str): Encrypted class name
            original_name (str): Original class name
        """
        encrypted_name = index_to_encrypted[label_index]
        original_name = encrypted_to_original[encrypted_name]
        return encrypted_name, original_name

    # Testing
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    # Detailed results
    results = []

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test data
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            # Get the probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Convert to numpy
            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()

            # Append to the lists
            all_probs.extend(probs_np)
            all_preds.extend(preds_np)
            all_labels.extend(labels_np)

            # Collect detailed results
            for i in range(len(labels_np)):
                pred_index = preds_np[i]
                label_index = labels_np[i]

                # Get class names
                pred_encrypted, pred_original = get_class_names(pred_index)
                label_encrypted, label_original = get_class_names(label_index)

                # Check if the prediction is correct
                is_correct = pred_index == label_index

                # Append to the results
                results.append({
                    'encrypted_class_name': label_encrypted,
                    'original_class_name': label_original,
                    'predicted_encrypted_class_name': pred_encrypted,
                    'predicted_original_class_name': pred_original,
                    'predicted_label': pred_index,
                    'actual_label': label_index,
                    'is_correct': is_correct
                })

    # Convert to numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # Compute accuracy
    accuracy = np.mean(all_preds_np == all_labels_np)

    # Compute precision, recall, and F1 score
    precision = precision_score(all_labels_np, all_preds_np, average='macro',
                                zero_division=0)
    recall = recall_score(all_labels_np, all_preds_np, average='macro',
                          zero_division=0)
    f1 = f1_score(all_labels_np, all_preds_np, average='macro',
                  zero_division=0)

    # Compute ROC AUC
    classes = range(num_classes)
    all_labels_bin = label_binarize(all_labels_np, classes=classes)

    # Compute ROC AUC score
    try:
        mean_auc = roc_auc_score(all_labels_bin, all_probs_np, average='macro',
                                 multi_class='ovr')
    except ValueError as e:
        print(f"ROC AUC calculation error: {e}")
        mean_auc = None

    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels_np, all_preds_np)

    # Log the results
    log_event(f"Test Accuracy: {accuracy:.4f}")
    log_event(f"Precision: {precision:.4f}")
    log_event(f"Recall: {recall:.4f}")
    log_event(f"F1 Score: {f1:.4f}")
    # Log ROC AUC if available
    if mean_auc is not None:
        log_event(f"ROC AUC: {mean_auc:.4f}")
    else:
        log_event("ROC AUC: NA")
    # Log confusion matrix
    log_event(f"Confusion Matrix:\n{conf_matrix}")

    # Save the detailed results to a CSV file
    test_results_csv = os.path.join(
        all_log_dir, "test_results.csv")

    # Save the results to a CSV file
    fieldnames = [
        'encrypted_class_name',
        'original_class_name',
        'predicted_encrypted_class_name',
        'predicted_original_class_name',
        'predicted_label',
        'actual_label',
        'is_correct'
    ]
    with open(test_results_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    log_event(f"Test results saved to {test_results_csv}")
