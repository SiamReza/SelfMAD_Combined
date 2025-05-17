from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import cv2
import torch
import logging

# Set up logging
def setup_logger():
    """Set up and return a logger that writes to both file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get the logger
    logger = logging.getLogger("dataset")

    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a file handler that appends to the log file
    file_handler = logging.FileHandler(os.path.join("logs", "dataset.log"), mode='a')
    file_handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize the logger
logger = setup_logger()

# ImageNet normalization parameters for ViT-MAE
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class MorphDataset(Dataset):
    def __init__(self, datapath, image_size, phase='train', transform=None, model_type=None):
        assert datapath is not None
        if "FF++" in datapath:
            assert phase in ['train','val','test']
            datapath = os.path.join(datapath, phase)

        labels = []
        image_paths = []
        for method in os.listdir(datapath):
            for root, _, files in os.walk(os.path.join(datapath, method)):
                if not files:
                    continue
                for filename in files:
                    if filename.endswith(('.png', '.jpg')):
                        image_paths.append(os.path.join(root, filename))
                        if "real" in root or "raw" in root:
                            labels.append(0)
                        else:
                            labels.append(1)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = image_size
        self.model_type = model_type

        logger.info(f"Initialized MorphDataset with {len(image_paths)} images, model_type={model_type}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.image_paths[idx]))
            label = self.labels[idx]
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.transpose((2,0,1))
            img = img.astype('float32')/255

            # Convert to PyTorch tensor
            img = torch.from_numpy(img)

            # Apply ImageNet normalization for ViT-MAE
            if self.model_type == "vit_mae_large":
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                img = (img - mean) / std

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.error(f"Error in MorphDataset.__getitem__ for index {idx}, image path: {self.image_paths[idx]}: {str(e)}")
            # Create a blank image as a fallback
            img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            return img, self.labels[idx]


class PartialMorphDataset(Dataset):
    def __init__(self, datapath, image_size, transform=None, method=None, model_type=None):
        assert datapath is not None
        assert method is not None
        if 'FRLL' in datapath:
            assert method in ['amsl', 'facemorpher', 'opencv', 'stylegan', 'webmorph']
            self.dataset_name = f"FRLL_{method}"
        elif 'FRGC' in datapath:
            assert method in ['facemorpher', 'opencv', 'stylegan']
            self.dataset_name = f"FRGC_{method}"
        elif 'FERET' in datapath:
            assert method in ['facemorpher', 'opencv', 'stylegan']
            self.dataset_name = f"FERET_{method}"

        labels = []
        image_paths = []
        for curr_method in os.listdir(datapath):
            for root, _, files in os.walk(os.path.join(datapath, curr_method)):
                if not files:
                    continue
                for filename in files:
                    if filename.endswith(('.png', '.jpg')) and method in root or "real" in root or "raw" in root:
                        image_paths.append(os.path.join(root, filename))
                        if "real" in root or "raw" in root:
                            labels.append(0)
                        else:
                            labels.append(1)

        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = image_size
        self.model_type = model_type

        logger.info(f"Initialized PartialMorphDataset with {len(image_paths)} images, model_type={model_type}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.image_paths[idx]))
            label = self.labels[idx]
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.transpose((2,0,1))
            img = img.astype('float32')/255

            # Convert to PyTorch tensor
            img = torch.from_numpy(img)

            # Apply ImageNet normalization for ViT-MAE
            if self.model_type == "vit_mae_large":
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                img = (img - mean) / std

            if self.transform:
                img = self.transform(img)

            return img, label
        except Exception as e:
            logger.error(f"Error in PartialMorphDataset.__getitem__ for index {idx}, image path: {self.image_paths[idx]}: {str(e)}")
            # Create a blank image as a fallback
            img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
            return img, self.labels[idx]

# class MorDIFF(Dataset):
#     def __init__(self, datapath_fake, datapath_real, image_size, transform=None):
#         assert datapath_fake is not None
#         assert datapath_real is not None

#         labels = []
#         image_paths = []
#         for curr_faces in os.listdir(datapath_fake):
#             for root, _, files in os.walk(os.path.join(datapath_fake, curr_faces)):
#                 if not files:
#                     continue
#                 for filename in files:
#                     if filename.endswith(('.png', '.jpg')):
#                         image_paths.append(os.path.join(root, filename))
#                         labels.append(1)

#         for filename in os.listdir(os.path.join(datapath_real, 'FRLL-Morphs_cropped', 'raw')):
#             if filename.endswith(('.png', '.jpg')):
#                 image_paths.append(os.path.join(datapath_real, 'FRLL-Morphs_cropped', 'raw', filename))
#                 labels.append(0)

#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform
#         self.img_size = image_size

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):

#         img=np.array(Image.open(self.image_paths[idx]))
#         label=self.labels[idx]
#         img = cv2.resize(img, (self.img_size, self.img_size))
#         img=img.transpose((2,0,1))
#         img = img.astype('float32')/255
#         if self.transform:
#             img = self.transform(img)
#         return img, label

import torch

class TestMorphDataset(Dataset):
    """Dataset for testing morph detection models"""
    def __init__(self, dataset_name, image_size=224, model_type=None):
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.model_type = model_type

        # Initialize lists
        self.image_paths = []
        self.labels = []

        # Get the base directory from environment or use a relative path
        # Try to use current directory's datasets folder first
        # Go up three levels from this file (SelfMAD-siam/utils/dataset.py) to reach the project root, then 'datasets'
        project_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = os.environ.get("DATASETS_DIR", os.path.join(project_root_guess, "datasets"))

        # Always load from test directories for proper evaluation
        # This ensures we use the full test dataset rather than a small subset
        print(f"Loading test data from directories for {dataset_name}")

        # Load bonafide test images (label 0)
        bonafide_path = os.path.join(base_dir, "bonafide", dataset_name, "test")
        if os.path.exists(bonafide_path):
            for img_file in os.listdir(bonafide_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(bonafide_path, img_file))
                    self.labels.append(0)

        # Load morph test images (label 1)
        morph_path = os.path.join(base_dir, "morph", dataset_name, "test")
        if os.path.exists(morph_path):
            for img_file in os.listdir(morph_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(morph_path, img_file))
                    self.labels.append(1)

        # Log the number of samples loaded
        bonafide_count = self.labels.count(0)
        morph_count = self.labels.count(1)
        print(f"Loaded {len(self.image_paths)} test samples from directories ({bonafide_count} bonafide, {morph_count} morph)")

        # If no images were found, log a debug message
        if len(self.image_paths) == 0:
            print(f"ERROR: No test images found for {dataset_name}. Please ensure test data exists in the following directories:")
            print(f"  - {bonafide_path}")
            print(f"  - {morph_path}")
            print(f"The model will not be able to evaluate properly without test data.")
            # Log to a file as well
            import datetime
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "missing_test_data.log")
            with open(log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] No test images found for {dataset_name}. Paths checked: {bonafide_path}, {morph_path}\n")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Load and preprocess image
            img_path = self.image_paths[idx]

            # Check if file exists - try both as is and relative to current directory
            if os.path.exists(img_path):
                try:
                    img = np.array(Image.open(img_path))
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    # Create a blank image as a fallback
                    img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
            else:
                # Try relative to SelfMAD-siam directory
                alt_path = os.path.join(".", img_path)
                if os.path.exists(alt_path):
                    try:
                        img = np.array(Image.open(alt_path))
                    except Exception as e:
                        print(f"Error loading image {alt_path}: {str(e)}")
                        # Create a blank image as a fallback
                        img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
                else:
                    print(f"Warning: Image file not found: {img_path} or {alt_path}")
                    # Create a blank image as a fallback
                    img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Resize to expected dimensions using Lanczos resampling
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)

            # Convert to PyTorch format
            img = img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Apply ImageNet normalization for ViT-MAE
            if self.model_type == "vit_mae_large":
                mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
                std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
                img = (img - mean) / std
                logger.info(f"Applied ImageNet normalization for ViT-MAE to image {idx}")

            # Return in the format expected by the evaluate function
            return img, self.labels[idx]
        except Exception as e:
            print(f"Unexpected error in __getitem__ for index {idx}: {str(e)}")
            # Create a blank image as a fallback
            img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            return torch.tensor(img), self.labels[idx]

def default_datasets(image_size, datasets="original", config=None, model_type=None):
    """
    Create default datasets for evaluation.

    Args:
        image_size: Size of the images
        datasets: Type of datasets to create ("original" or "custom_morph")
        config: Configuration dictionary with dataset paths
        model_type: Type of model being used (e.g., "vit_mae_large")

    Returns:
        Dictionary of datasets
    """
    assert datasets in ["original", "custom_morph"]
    logger.info(f"Creating default datasets with model_type={model_type}")

    if datasets == "original":
        FRGC_datapath = config["FRGC_path"]
        FERET_datapath = config["FERET_path"]
        FRLL_datapath = config["FRLL_path"]

        test_datasets = {
            "FRGC": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='facemorpher', model_type=model_type),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='opencv', model_type=model_type),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='stylegan', model_type=model_type)
            },
            "FERET": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='facemorpher', model_type=model_type),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='opencv', model_type=model_type),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='stylegan', model_type=model_type)
            },
            "FRLL": {
                "amsl": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='amsl', model_type=model_type),
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='facemorpher', model_type=model_type),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='opencv', model_type=model_type),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='stylegan', model_type=model_type),
                "wm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='webmorph', model_type=model_type)
            }
        }
        return test_datasets
    elif datasets == "custom_morph":
        # Create test datasets for our custom morph datasets
        morph_datasets = ["LMA", "LMA_UBO", "MIPGAN_I", "MIPGAN_II", "MorDiff", "StyleGAN"]
        test_datasets = {}

        for dataset_name in morph_datasets:
            # Check if dataset path is in config
            path_key = f"{dataset_name}_path"
            if path_key in config and config[path_key] is not None:
                # Create test dataset
                test_datasets[dataset_name] = {
                    "test": TestMorphDataset(dataset_name=dataset_name, image_size=image_size, model_type=model_type)
                }

        return test_datasets