import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from photo_augmentation import BanknoteDataset

# to fix multiprocessing issue for MACOS (Optional)
if __name__ == '__main__':
    import platform
    if platform.system() == 'Darwin':
        import multiprocessing
        multiprocessing.set_start_method('fork')  # Fixes macOS multiprocessing

# image transformations for training data with augmentation
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256
    transforms.RandomHorizontalFlip(),  # Randomly flip image horizontally
    transforms.RandomRotation(15),  # Random rotation up to 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Random color changes
    transforms.RandomResizedCrop(224),  # Random crop then resize to 224x224
    transforms.ToTensor(),  # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std
])

# Define transformations for validation/test data with no augmentation
val_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resizing image
    transforms.CenterCrop(224),  # Crop to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                         std=[0.229, 0.224, 0.225])  # Normalize using ImageNet std
])

def main():
    dataset_path = "dataset"  # Path to the dataset folder
    full_dataset = BanknoteDataset(root_dir=dataset_path, transform=None)  # Load dataset without transforms

    # Split dataset into train validation and testing dataset
    train_size = int(0.7 * len(full_dataset))  # 70% for training
    val_size = int(0.15 * len(full_dataset))  # 15% for validation
    test_size = len(full_dataset) - train_size - val_size  # remainder used for testing

    # Perform the split
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

    train_set.dataset.transform = train_transform  # Apply training transforms
    val_set.dataset.transform = val_transform  # Apply validation transforms
    test_set.dataset.transform = val_transform  # Apply validation transforms (same as val)

    # Creating DataLoaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)

    # Verify dataset
    print(f"Total images: {len(full_dataset)}")
    print(f"Training set: {len(train_set)} images")
    print(f"Validation set: {len(val_set)} images")
    print(f"Test set: {len(test_set)} images")

    # Optional: Visualize a batch
    def imshow(img):
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean  # Un-normalize
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.axis('off')

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    fig = plt.figure(figsize=(12, 6))
    for idx in range(6):  # Show 6 samples
        ax = fig.add_subplot(2, 3, idx+1)
        imshow(images[idx])
        ax.set_title(full_dataset.label_encoder.inverse_transform([labels[idx]])[0])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()