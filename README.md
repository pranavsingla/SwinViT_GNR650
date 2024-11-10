# Personalised Remote Sensing Image Classification Library

## Overview
This library facilitates the classification of optical remote sensing images using advanced deep learning models such as Vision Transformer (ViT), Swin Transformer (SwinViT), etc... This library also facilitates the automatic downloading of remote sensing datasets like Million AID and arranges them in the directory structure as shown below. It’s designed to be modular and extendable, enabling easy integration of additional datasets, models, and training strategies.

## Table of Contents
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Adding New Datasets and Models](#adding-new-datasets-and-models)
- [Contributing](#contributing)
- [Deployment](#deployment)

---

## Setup

1. **Clone the repository:**
```bash
git clone https://github.com/pranavsingla/SwinViT_GNR650.git
cd remote-sensing-library
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Dataset Preparation:**
The dataset manager supports any directory structure compatible with PyTorch’s ImageFolder. Organize images into folders by class name under a main directory.
Example:
```kotlin
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

## Usage
### Running Training
To train a model, run the following command:
```bash
python main.py --config path/to/config.yaml
```
### Example Configuration File (config.yaml)
Below is an example configuration file, which you can modify based on your training requirements:
```yaml
dataset:
  data_dir: "path/to/dataset"
  batch_size: 32

model:
  name: "swinvit"
  pretrained_model: "microsoft/swin-tiny-patch4-window7-224"

training:
  epochs: 20
  lr: 0.001
  device: "cuda"
```
* ```dataset```: Details on dataset location and parameters.
* ```model```: Choose between "vit" or "swinvit", and specify a pretrained model path if desired.
* ```training```: Define the number of epochs, learning rate, and the computing device.

## Adding New Datasets and Models
### Adding a New Dataset
To add a new dataset:

1. Create a new Python file in the ```datasets/``` folder.
2. Implement a data-loading class similar to ```DatasetManager``` in ```datasets/million_aid.py```.
3. Add any dataset-specific preprocessing as needed.

### Adding a New Model
1. Create a new Python file in the ```models/``` folder for the model architecture.
2. Implement a class that initializes the model and defines its forward pass.
3. Import and register the model in ```models/__init__.py```.

### Contributing
We welcome contributions to enhance the functionality of this library. To contribute:

1. Fork the repository and clone your forked version locally.
2. Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/new-feature
```
3. Add your changes in the relevant files. Ensure new features are well-documented and modular.
4. Run tests to validate your changes (test scripts coming soon).
5. Commit your changes and push to your forked repository:
```bash
git add .
git commit -m "Add feature/new-feature"
git push origin feature/new-feature
```
6. Open a pull request to the main repository.

## Coding Standards
* Follow the structure of existing modules for consistency.
* Write clear, concise comments and docstrings for new functions and classes.
* Ensure new code adheres to PEP8 standards.

## Reporting Issues
If you encounter any bugs or have suggestions, please open an issue on GitHub.

## Deployment
### Packaging the Library
To package and distribute the library:

1. Modify the ```setup.py ```(add if missing) to define the package information and dependencies.
2. Run the following command to build the package:
```bash
python setup.py sdist bdist_whee
```
3. Use PyPI or a private repository to distribute.

### Deployment in a Cloud Environment
To deploy in a cloud environment like AWS or GCP:

1. Create a cloud VM with a GPU if training on a large dataset.
2. Set up the environment by following the Setup steps.
3. Upload your dataset to cloud storage or directly to the VM.
4. Run training by executing:
```bash
nohup python main.py --config path/to/config.yaml &
```
This runs the training in the background, enabling you to log out without interrupting the process.

### Docker Setup
1. Build a Dockerfile that includes necessary dependencies, GPU support, and entry points for training.
2. Build and run the Docker container with:
```bash
docker build -t remote-sensing-library .
docker run --gpus all -v $(pwd)/data:/data -it remote-sensing-library
```
## License
This project is licensed under the MIT License. See the LICENSE file for details.