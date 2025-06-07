import os  # To handle file path
from PIL import Image  # To open image files
from torch.utils.data import Dataset  # Base class for PyTorch datasets
from sklearn.preprocessing import LabelEncoder  # For encoding labels into numbers

# Custom dataset class for loading banknote images
class BanknoteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Root directory to the class folder
        self.transform = transform  # Image transformations for augmentation and transformation

        self.image_paths = []  # List to store paths to all images
        self.labels = []  # Corresponding labels (folder names)

        # Get sorted list of class names excluding hidden one
        self.class_names = [f for f in sorted(os.listdir(root_dir)) if not f.startswith('.')]

        # Encoding labels into numerical value
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.class_names)

        # Loop over each class folder
        for folder in self.class_names:
            folder_path = os.path.join(root_dir, folder)  # Full path to folder
            if os.path.isdir(folder_path):  # Ensure it's a directory
                for file in os.listdir(folder_path):  # Loop through files in the folder
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')) and not file.startswith('.'):
                        self.image_paths.append(os.path.join(folder_path, file))  # Save full image path
                        self.labels.append(folder)  # Save label as folder name

        # Encode all labels to integers
        self.encoded_labels = self.label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.image_paths)  # Return total number of images

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')  # Load image and convert to RGB
        label = self.encoded_labels[idx]  # Get the label
        if self.transform:
            img = self.transform(img)  # Apply transformations
        return img, label  # Return image and numerical labels
