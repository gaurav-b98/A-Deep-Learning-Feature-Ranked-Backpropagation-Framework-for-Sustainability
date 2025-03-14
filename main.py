# System imports
import os
# Third-party imports
import yaml
import argparse
import torch.multiprocessing as mp
# Local imports
from train import train
from test import test_model


def main():
    """
    Main function to start training and testing processes
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and Test Models')
    # Add arguments - config file, generate graphs
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--generate_graphs', action='store_true',
                        help='Generate graphs after training and testing')

    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_type = config['model']['model_type']
    architecture = config['model']['architecture']
    # Set environment variables for distributed training
    world_size = config['training']['world_size']

    os.environ['MASTER_ADDR'] = config['training']['master_addr']
    os.environ['MASTER_PORT'] = config['training']['master_port']

    # Start training processes
    mp.spawn(train, args=(config,), nprocs=world_size, join=True)
    # Ensure all training processes have completed
    for p in mp.active_children():
        p.join()

    print("Training processes has completed.")

    # Run testing (on a single GPU)
    test_model(args.config)
    print("Testing processes has completed.")

    # Generate graphs
    if args.generate_graphs:
        from utils.graphs import plot_metrics
        if architecture == "standard":
            metrics_csv = os.path.join(
                config['logging']['log_dir'],
                f"{model_type}_{architecture}/metrics.csv")
            graphs_dir = os.path.join(config['logging']['graphs_dir'],
                                      f"{model_type}_{architecture}")
        elif architecture == "selective":
            schedule = config['training']['percentage_schedule']
            metrics_csv = os.path.join(
                config['logging']['log_dir'],
                f"{model_type}_{architecture}_{schedule}/metrics.csv")
            graphs_dir = os.path.join(
                config['logging']['graphs_dir'],
                f"{model_type}_{architecture}_{schedule}")
        plot_metrics(metrics_csv, graphs_dir)


if __name__ == '__main__':
    main()
