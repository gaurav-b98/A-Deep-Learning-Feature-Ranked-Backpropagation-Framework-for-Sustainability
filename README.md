# A Deep Learning Feature-Ranked Backpropagation Framework for Sustainability

## Overview
This research project introduces a novel deep learning framework that integrates feature-ranking into the backpropagation process to improve training efficiency and reduce energy consumption. The proposed framework dynamically prioritizes the most critical weights for updates, leading to faster training and lower computational costs while maintaining model performance. The implementation is based on AlexNet, ResNet-18, and VGGNet-19, trained on a subset of the ImageNet dataset.

## Features
- **Feature-Ranked Backpropagation**: Prioritizes significant weight updates using gradient-based saliency maps.
- **Selective Weight Updates**: Reduces unnecessary computations, improving training efficiency.
- **Support for Multiple Models**: Works with AlexNet, ResNet-18, and VGGNet-19.
- **Configurable Training Parameters**: Allows customization through `config.yaml`.
- **Logging & Visualization**: Tracks training metrics, energy consumption, and logs results in CSV and graph formats.
- **Energy Efficiency**: Reduces training time by up to 15% and energy consumption by up to 20%.

## System Requirements
### Hardware
- **CPU**: 16-core or higher (multi-core processing preferred)
- **GPU**: At least 1 NVIDIA GPU with 16GB VRAM (Tesla T4 or equivalent)
- **RAM**: Minimum 16GB
- **Storage**: At least 500GB free disk space

### Software
- **Operating System**: Ubuntu 22.04 or Windows equivalent
- **GPU Driver**: NVIDIA Driver version 525 or higher
- **CUDA Toolkit**: Version 11.8 or higher
- **cuDNN**: Version 8.7 or higher
- **Python**: Version 3.8+
- **PyTorch**: Version 2.0+

## Installation
### 1. Clone the Repository
```sh
 git clone https://github.com/yourusername/deep-learning-feature-ranked-backpropagation.git
 cd deep-learning-feature-ranked-backpropagation
```

### 2. Create a Conda Environment
```sh
 conda env create -n feature_ranked_bp -f environment.yml
 conda activate feature_ranked_bp
```

### 3. Prepare the Dataset
The dataset used is the ImageNet-1k subset. Download it from HuggingFace using the script below:
```sh
 wget --header="Authorization: Bearer <access_token>" \ 
 https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/<file_name>.tar.gz \ 
 -O <file_name>.tar.gz
```
Extract the dataset:
```sh
 tar -xzvf <file_name>.tar.gz
```
Organize dataset using the provided scripts:
```sh
 python scripts/move_train_files.py
 python scripts/move_val_files.py
```
Split the dataset:
```sh
 python scripts/split_dataset.py --train 60 --val 20 --test 20
```

## Configuration
Edit the `config.yaml` file to customize experimental settings:
```yaml
model:
  type: "resnet" # Options: alexnet, resnet, vggnet
  num_classes: 50
  architecture: "selective" # Options: standard, selective
training:
  learning_rate: 0.0001
  num_epochs: 50
  batch_size: 32
  high_percentage: 50
  low_percentage: 20
  percentage_schedule: "linear" # Options: linear, exponential
logging:
  log_dir: "logs"
  log_interval: 10
```

## Running the Experiments
### 1. Train and Test
```sh
 python main.py --config config.yaml
```
For additional metrics visualization:
```sh
 python main.py --config config.yaml --generate_graphs
```

## Results & Evaluation
- Training and validation metrics are logged in `logs/`.
- Test results are saved in `logs/`.
- Model weights are stored in `checkpoints/`.
- Energy consumption is monitored using `codecarbon`.
- Training efficiency metrics include accuracy, training time, energy usage, and CO2 emissions.

## Citation
If you use this research, please cite:
```
@article{Gaurav2024,
  title={A Deep Learning Feature-Ranked Backpropagation Framework for Sustainability},
  author={Gaurav},
  year={2024},
  institution={National College of Ireland}
}
```

## License
This project is licensed under the MIT License.

## Acknowledgments
This research was conducted as part of the MSc in Artificial Intelligence at the National College of Ireland, under the supervision of Paul Stynes.
