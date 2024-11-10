import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

class DatasetManager:
    def __init__(self, data_dir, transform=None, batch_size=32, train_split=0.8):
        self.data_dir = data_dir  # data_dir should be 'data/million_aid'
        self.batch_size = batch_size
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.train_split = train_split

    def load_data(self):
        # Now, datasets.ImageFolder will load from train and val subdirectories under data_dir
        train_dir = os.path.join(self.data_dir, 'train')  # 'data/million_aid/train'
        val_dir = os.path.join(self.data_dir, 'val')  # 'data/million_aid/val'

        # Load training dataset and validation dataset from respective directories
        train_dataset = datasets.ImageFolder(train_dir, transform=self.transform)
        val_dataset = datasets.ImageFolder(val_dir, transform=self.transform)

        # Split the train dataset if required
        train_size = int(self.train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_data, val_data = random_split(train_dataset, [train_size, val_size])

        # Create DataLoader for both train and validation sets
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader
