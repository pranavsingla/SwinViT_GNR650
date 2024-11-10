# main.py

import torch
import argparse
from local_datasets import DatasetManager
from models import SwinViT
from training import train_model
from utils import load_config, download_and_organize_million_aid #download_million_aid
# from utils.dataset_utils import download_million_aid

def main():
    # Parse arguments to get the path to the config file
    parser = argparse.ArgumentParser(description="Remote Sensing Image Classification")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    # Load the configuration from the YAML file
    config = load_config(args.config)

    # Download the dataset based on the configuration in the YAML file
    dataset_dir = config['dataset']['data_dir']  # Get the dataset directory from config
    train_dataset = download_and_organize_million_aid(dataset_name="jonathan-roberts1/Million-AID", data_dir=dataset_dir)
    # train_dataset = download_million_aid(dataset_name="jonathan-roberts1/Million-AID", download_dir=dataset_dir)

    # Load the dataset into DataLoader here (assuming DatasetManager uses this format)
    dataset_manager = DatasetManager(dataset_dir, batch_size=config['dataset']['batch_size'])  # Use the same directory as in the config
    train_loader, val_loader = dataset_manager.load_data()  # Adjust this according to your DatasetManager logic
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Load the model
    model = SwinViT(config['model'])

    # Set up the optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['scheduler']['step_size'], gamma=config['training']['scheduler']['gamma'])

    # Train the model
    trained_model = train_model(config, model, dataloaders, criterion, optimizer, scheduler, config['training']['epochs'])

if __name__ == "__main__":
    main()
