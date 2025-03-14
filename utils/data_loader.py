# Imports
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def get_data_loaders(config, rank=0, world_size=1):
    """
    Function to load data and create DataLoaders
    Args:
        config (dict): Configuration dictionary
        rank (int): Rank of the process
        world_size (int): Total number of processes
    Returns:
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader): DataLoader for validation data
        test_loader (DataLoader): DataLoader for test data
    """
    # Define data augmentations
    augmentations = []
    # Define augmentations based on config
    augmentations.append(transforms.RandomResizedCrop(224))
    if config['data']['augmentations']['horizontal_flip']:
        augmentations.append(transforms.RandomHorizontalFlip())

    if config['data']['augmentations']['rotation'] > 0:
        augmentations.append(transforms.RandomRotation(
            config['data']['augmentations']['rotation']))

    if config['data']['augmentations']['color_jitter']:
        augmentations.append(transforms.ColorJitter())

    if config['data']['augmentations']['random_crop']:
        augmentations.append(transforms.RandomCrop(224))

    # Normalize the images
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    augmentations.append(transforms.ToTensor())
    augmentations.append(normalize)

    # Remove None values in case any augmentations are disabled
    train_transforms = transforms.Compose(
        [aug for aug in augmentations if aug is not None])

    val_test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        config['data']['train_dir'], transform=train_transforms)

    val_dataset = datasets.ImageFolder(
        config['data']['val_dir'], transform=val_test_transforms)

    test_dataset = datasets.ImageFolder(
        config['data']['test_dir'], transform=val_test_transforms)

    # Create samplers for DDP
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['model']['batch_size'],
        sampler=train_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['model']['batch_size'],
        sampler=val_sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    """
    Note: The test loader is not distributed as we do not need to
          evaluate the model in a distributed manner.
    """
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['model']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
