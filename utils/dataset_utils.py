import os
from datasets import load_dataset
from shutil import copyfile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image

def download_and_organize_million_aid(dataset_name="jonathan-roberts1/Million-AID", data_dir="data"):
    """
    Downloads the Million-AID dataset from Hugging Face, saves the images to the specified directory structure
    for ImageFolder to read, and organizes them into train and validation sets.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face.
        data_dir (str): The directory where the dataset will be saved.

    Returns:
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
    """
    # Create the necessary directories for the ImageFolder structure
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Download the dataset
    print("Downloading Million-AID dataset...")
    dataset = load_dataset(dataset_name)

    # print(dataset, '\n\n', dataset['train'], '\n\n', dataset.keys(), '\n\n')
    # # Let's print an example from the dataset
    # example = dataset["train"][0]  # Assuming "train" split exists
    # print(example)  # Print the example to inspect its format

    # class_name = example["label_1"]
    # print(f"Class name: {class_name}", '\n\n\n\n')

    # Create class subdirectories under train and val
    if os.path.exists(os.path.join(train_dir, str(0))):
        print( "Dataset already downloaded and organized.")
        return "Dataset already downloaded and organized."
    for split in ["train"]: #, "test"]:
        split_data = dataset[split]
        # print(split_data,'\n\n\n', len(split_data), "\n\n\n")

        # Split into train and validation data (80-20 split)
        num_train = int(len(split_data) * 0.8)
        num_val = len(split_data) - num_train
        train_data, val_data = split_data[:num_train], split_data[num_train:]
        # print(len(train_data['label_1']), "\n\n\n", train_data['label_2'],'\n\n\n')

        # Organize images into class directories
        for idx, example in enumerate(train_data['image']):
            example = {'image' : train_data['image'][idx], 
                       'label_1' : train_data['label_1'][idx], 
                       'label_2' : train_data['label_2'][idx], 
                       'label_3' : train_data['label_3'][idx]}
            for i in range(1,4):
                class_name = example[f"label_{i}"]
                class_dir = os.path.join(train_dir, str(class_name))
                os.makedirs(class_dir, exist_ok=True)
                image_path = example["image"]
                new_image_path = os.path.join(class_dir, f"{idx}.jpg")
                image_path.save(new_image_path)

        for idx, example in enumerate(val_data['image']):
            example = {'image' : train_data['image'][idx], 
                       'label_1' : train_data['label_1'][idx], 
                       'label_2' : train_data['label_2'][idx], 
                       'label_3' : train_data['label_3'][idx]}
            for i in range(1,4):
                class_name = example[f"label_{i}"]
                class_dir = os.path.join(val_dir, str(class_name))
                os.makedirs(class_dir, exist_ok=True)
                image_path = example["image"]
                new_image_path = os.path.join(class_dir, f"{idx}.jpg")
                image_path.save(new_image_path)

    print(f"Dataset organized in {data_dir}")

    # Now load the data using ImageFolder
    return "loaded dataset and organised and images saved" #load_data(data_dir)


def load_data(data_dir, batch_size=32, train_split=0.8):
    """
    Load the dataset using ImageFolder for both training and validation sets.
    
    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for the data loaders.
        train_split (float): The fraction of the dataset to use for training.
    
    Returns:
        train_loader: DataLoader for the training data.
        val_loader: DataLoader for the validation data.
    """
    # Define the necessary transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split the dataset into training and validation sets
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    # Create DataLoader for train and validation datasets
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    download_and_organize_million_aid()









# # utils/dataset_utils.py

# import os
# from datasets import load_dataset

# def download_million_aid(dataset_name="jonathan-roberts1/Million-AID", download_dir="data"):
#     """
#     Downloads the Million-AID dataset from Hugging Face and organizes it into the required directory structure.
#     If the dataset already exists, it will skip the download.

#     Args:
#         dataset_name (str): Name of the dataset on Hugging Face.
#         download_dir (str): Directory where the dataset will be saved.

#     Returns:
#         train_dataset: The loaded training dataset.
#     """
#     # Create directory structure if it doesn't exist
#     if not os.path.exists(download_dir):
#         os.makedirs(download_dir)
    
#     dataset_path = os.path.join(download_dir, "Million-AID")
    
#     # Check if dataset is already downloaded
#     if os.path.exists(dataset_path):
#         print(f"Dataset already exists at {dataset_path}. Loading it...")
#         return load_dataset(dataset_path)["train"]  # Load from disk if already downloaded

#     print("Downloading Million-AID dataset...")
#     dataset = load_dataset(dataset_name)
    
#     # Save the dataset to disk
#     print("Saving dataset to disk...")
#     dataset["train"].save_to_disk(dataset_path)  # Save the train split

#     print(f"Dataset downloaded and saved at {dataset_path}")
#     return dataset["train"]



# # Example usage:
# if __name__ == "__main__":
#     download_million_aid()
