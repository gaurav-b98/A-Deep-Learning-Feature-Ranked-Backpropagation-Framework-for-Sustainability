# System Imports
import os
import time
from datetime import timedelta
# Third-party Imports
import csv
import psutil
import pynvml
import random
import numpy as np
from codecarbon import EmissionsTracker
# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
# Custom Imports
from utils.logging import setup_logger, log_event
from utils.data_loader import get_data_loaders
# Import models
from models.resnet import ResNetStandard, ResNetSelective
from models.vggnet import VGGNetStandard, VGGNetSelective
from models.alexnet import AlexNetStandard, AlexNetSelective
# Import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def train(rank, config):
    """
    Train the model using the training data loader and log the results
    Args:
        rank (int): Rank of the process
        config (dict): Configuration dictionary
    """
    try:
        # Initialize pynvml
        pynvml.nvmlInit()

        # Load the configuration file
        log_dir = config['logging']['log_dir']
        model_type = config['model']['model_type']
        architecture = config['model']['architecture']
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

        # Only rank 0 will create the directory
        if rank == 0:
            # Create directory for Emissions logs
            os.makedirs(all_log_dir, exist_ok=True)
            tracker = EmissionsTracker(output_dir=all_log_dir)
            tracker.start()
            total_training_start_time = time.time()
            # Create metrics log file
            metrics_log_file = os.path.join(all_log_dir, "metrics.csv")
            with open(metrics_log_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'epoch', 'train_loss', 'train_accuracy', 'val_loss',
                    'val_accuracy', 'cpu_usage', 'gpu_usage', 'ram_usage',
                    'epoch_time', 'emissions', 'precision', 'recall',
                    'f1_score', 'auc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
            # Setup logger
            setup_logger(all_log_dir, "train.log")
            # Create model directory
            os.makedirs(model_dir, exist_ok=True)

        # Set up environment variables for DDP
        world_size = config['training']['world_size']
        backend = config['training']['backend']

        # Initialize process group
        dist.init_process_group(
            backend=backend, init_method='env://', world_size=world_size,
            rank=rank, timeout=timedelta(minutes=30))

        # Set device
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)

        # Get data loaders
        train_loader, val_loader, _ = get_data_loaders(
            config, rank=rank, world_size=world_size)

        # Extract configuration parameters
        num_classes = config['model']['num_classes']
        high_percentage = config['training']['high_percentage']
        low_percentage = config['training']['low_percentage']
        percentage_schedule = config['training']['percentage_schedule']
        num_epochs = config['training']['num_epochs']

        # Model selection
        if model_type == 'alexnet':
            if architecture == 'standard':
                model = AlexNetStandard(num_classes=num_classes)
            elif architecture == 'selective':
                model = AlexNetSelective(
                    num_classes=num_classes,
                    high_percentage=high_percentage,
                    low_percentage=low_percentage,
                    num_epochs=num_epochs,
                    schedule=percentage_schedule
                )
            else:
                raise ValueError(f"Invalid architecture: {architecture}")
        elif model_type == 'resnet':
            if architecture == 'standard':
                model = ResNetStandard(num_classes=num_classes)
            elif architecture == 'selective':
                model = ResNetSelective(
                    num_classes=num_classes,
                    high_percentage=high_percentage,
                    low_percentage=low_percentage,
                    num_epochs=num_epochs,
                    schedule=percentage_schedule
                )
            else:
                raise ValueError(f"Invalid architecture: {architecture}")
        elif model_type == 'vggnet':
            if architecture == 'standard':
                model = VGGNetStandard(num_classes=num_classes)
            elif architecture == 'selective':
                model = VGGNetSelective(
                    num_classes=num_classes,
                    high_percentage=high_percentage,
                    low_percentage=low_percentage,
                    num_epochs=num_epochs,
                    schedule=percentage_schedule
                )
            else:
                raise ValueError(f"Invalid architecture: {architecture}")
        else:
            raise ValueError(f"Invalid model type: {model_type}")

        model.to(device)

        # Wrap the model with DDP
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=config['training']['learning_rate'])

        # Log model and optimizer details
        if rank == 0:
            log_event("Using optimizer: Adam")
            log_event(
                f"Training model: {model_type}, Architecture: {architecture}")

        best_val_accuracy = 0.0

        # Early stopping parameters
        early_stopping_enabled = \
            config['training']['early_stopping']['enabled']
        patience = config['training']['early_stopping']['patience']
        epochs_no_improve = 0
        min_val_loss = np.inf

        # Training loop
        for epoch in range(num_epochs):
            if rank == 0:
                log_event(f"Epoch {epoch+1}/{num_epochs}")
                epoch_start_time = time.time()

            # Set epoch for sampler
            train_loader.sampler.set_epoch(epoch)

            # Training phase
            model.train()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Zero the parameter gradients
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward + optimize
                loss.backward()

                # Selective weight update
                if architecture == 'selective':
                    model.module.apply_selective_weight_update(epoch)
                # Update weights
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                running_corrects += torch.sum(preds == labels.data).item()

                # Log statistics (only on rank 0)
                if rank == 0 and \
                        (i + 1) % config['logging']['log_interval'] == 0:
                    batch_loss = running_loss / ((i + 1) * inputs.size(0))
                    batch_accuracy = \
                        running_corrects / ((i + 1) * inputs.size(0))
                    log_event(
                        f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")

            # Gather training results from all ranks
            total_loss = torch.tensor(running_loss).to(device)
            total_corrects = torch.tensor(running_corrects).to(device)
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(total_corrects, dst=0, op=dist.ReduceOp.SUM)

            # Validation phase
            model.eval()
            val_running_loss = 0.0
            val_running_corrects = 0
            all_probs = []
            all_labels = []

            # Disable gradient computation
            with torch.no_grad():
                # Iterate over validation data
                for data in val_loader:
                    inputs, labels = data
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_running_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(outputs, dim=1)
                    val_running_corrects += \
                        torch.sum(preds == labels.data).item()

                    # Get probabilities and labels for ROC AUC
                    probs = torch.softmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            # Gather validation results from all ranks
            val_total_loss = torch.tensor(val_running_loss).to(device)
            val_total_corrects = torch.tensor(val_running_corrects).to(device)
            dist.reduce(val_total_loss, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(val_total_corrects, dst=0, op=dist.ReduceOp.SUM)

            # Log statistics (only on rank 0)
            if rank == 0:
                epoch_loss = total_loss.item() / (len(train_loader.dataset))
                epoch_accuracy = \
                    total_corrects.item() / (len(train_loader.dataset))
                val_loss = \
                    val_total_loss.item() / (len(val_loader.dataset))
                val_accuracy = \
                    val_total_corrects.item() / (len(val_loader.dataset))

                log_event(
                    f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
                log_event(
                    f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

                # Compute metrics
                all_probs_np = np.array(all_probs)
                all_labels_np = np.array(all_labels)
                preds = np.argmax(all_probs_np, axis=1)
                # Compute precision, recall, and F1 Score
                precision = precision_score(
                    all_labels_np, preds, average='macro', zero_division=0)
                recall = recall_score(
                    all_labels_np, preds, average='macro', zero_division=0)
                f1 = f1_score(
                    all_labels_np, preds, average='macro', zero_division=0)

                # Compute ROC AUC Score
                classes = range(num_classes)
                all_labels_bin = label_binarize(all_labels_np, classes=classes)

                try:
                    mean_auc = roc_auc_score(
                        all_labels_bin, all_probs_np, average='macro',
                        multi_class='ovr')
                except ValueError as e:
                    print(f"ROC AUC calculation error: {e}")
                    mean_auc = None

                # Record end time for epoch
                epoch_time = time.time() - epoch_start_time

                # Get resource usage
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().percent
                handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
                gpu_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

                # Log metrics to CSV
                with open(metrics_log_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({
                        'epoch': epoch + 1,
                        'train_loss': epoch_loss,
                        'train_accuracy': epoch_accuracy,
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                        'cpu_usage': cpu_usage,
                        'gpu_usage': gpu_usage,
                        'ram_usage': ram_usage,
                        'epoch_time': epoch_time,
                        'emissions': tracker,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'auc': mean_auc if mean_auc is not None else 'NA'
                    })

                # Early stopping
                if early_stopping_enabled:
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
                        epochs_no_improve = 0
                        # Save the best model
                        model_save_path = os.path.join(
                            model_dir, "model_best.pth")

                        torch.save(model.module.state_dict(), model_save_path)
                    else:
                        epochs_no_improve += 1
                        # Check if early stopping criteria is met
                        if epochs_no_improve >= patience:
                            log_event(f"Early stopping triggered after {epoch+1} epochs.")
                            # Stop training
                            break
                else:
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        # Save the best model
                        model_save_path = os.path.join(
                            model_dir, "model_best.pth")
                        torch.save(model.module.state_dict(), model_save_path)

                # Reset start time for next epoch
                epoch_start_time = time.time()

        # Stop the emissions tracker
        if rank == 0:
            total_emissions = tracker.stop()
            total_training_time = time.time() - total_training_start_time
            log_event(
                f"Total Training Time: {total_training_time:.2f} seconds")
            log_event(
                f"Total Emissions: {total_emissions:.6f} kg CO2eq")
            log_event("Training complete!")

    except Exception as e:
        print(f"Rank {rank}: Exception occurred - {e}")

    finally:
        # Destroy process group
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception as e:
                print(f"Rank {rank}: Exception during destroy_process_group - {e}")
        # Shutdown pynvml
        pynvml.nvmlShutdown()
        print(f"Rank {rank}: Exiting training.")
