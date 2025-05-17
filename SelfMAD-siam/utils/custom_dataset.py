import os
import torch
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import cv2
import logging
from utils.selfMAD import selfMAD_Dataset

# Set up logging
def setup_logger():
    """Set up and return a logger that writes to both file and console."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Get the logger
    logger = logging.getLogger("custom_dataset")

    # Clear any existing handlers to avoid duplicate logs
    if logger.handlers:
        logger.handlers.clear()

    # Set the logging level
    logger.setLevel(logging.INFO)

    # Create a file handler that appends to the log file
    file_handler = logging.FileHandler(os.path.join("logs", "custom_dataset.log"), mode='a')
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

class MorphDataset(selfMAD_Dataset):
    def __init__(self, dataset_name, phase='train', image_size=224, train_val_split=0.8, csv_path=None):
        """
        Custom dataset adapter for morph detection that extends selfMAD_Dataset

        Args:
            dataset_name: Name of the dataset (LMA, LMA_UBO, etc.)
            phase: 'train', 'val', or 'test'
            image_size: Size of the images
            train_val_split: Ratio of training data (0.8 = 80% train, 20% val)
            csv_path: Path to save/load CSV file with image paths and labels
        """
        # Initialize transforms from parent class first
        super().__init__(phase=phase, image_size=image_size, datapath=None)

        self.dataset_name = dataset_name
        self.train_val_split = train_val_split

        # Ensure image_size is properly handled
        if isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)

        # Base paths - use os.path.join for cross-platform compatibility
        # Try multiple possible locations for datasets
        possible_base_dirs = [
            os.environ.get("DATASETS_DIR"),  # First check environment variable
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"),  # Check relative to script
            os.path.join(".", "datasets"),  # Check in current directory
            os.path.join("..", "datasets"),  # Check one level up
            "/cluster/home/aminurrs/SelfMAD_Combined/datasets",  # Check specific cloud path
            os.path.abspath("datasets")  # Check absolute path
        ]

        # Use the first directory that exists, or default to the second option
        project_root_guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                        os.path.join(project_root_guess, "datasets"))

        print(f"MorphDataset using datasets directory: {base_dir}")

        self.bonafide_path = os.path.join(base_dir, "bonafide", dataset_name)
        self.morph_path = os.path.join(base_dir, "morph", dataset_name)

        # Print paths for debugging
        print(f"Bonafide path: {self.bonafide_path}")
        print(f"Morph path: {self.morph_path}")

        # Initialize lists for images and labels
        self.image_list = []
        self.labels = []
        self.path_lm = []  # For landmarks
        self.face_labels = []  # For face labels

        # CSV handling
        self.csv_path = csv_path
        if csv_path is not None and os.path.exists(csv_path) and phase != 'test':
            # Load from existing CSV
            self.load_from_csv(csv_path)
        else:
            # Create new dataset
            self.create_dataset()
            if csv_path is not None and phase != 'test':
                self.save_to_csv(csv_path)

    def create_dataset(self):
        """Create dataset from folder structure"""
        if self.phase == 'test':
            # For test phase, use the 'test' folder
            folder_name = 'test'
        else:
            # For train/val phase, use the 'train' folder
            folder_name = 'train'

        # Load all images first, then split into train/val
        all_images = []
        all_labels = []
        all_path_lm = []
        all_face_labels = []

        # Load bonafide images (label 0)
        bonafide_folder = os.path.join(self.bonafide_path, folder_name)
        bonafide_folder = self.normalize_path(bonafide_folder)
        print(f"Looking for bonafide images in: {bonafide_folder}")

        if os.path.exists(bonafide_folder):
            for img_file in os.listdir(bonafide_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(bonafide_folder, img_file)
                    img_path = self.normalize_path(img_path)  # Normalize path
                    all_images.append(img_path)
                    all_labels.append(0)  # Bonafide label

                    # Add placeholder paths for landmarks and face labels
                    # These will be generated on-the-fly during training
                    all_path_lm.append(None)
                    all_face_labels.append(None)

        # Load morph images (label 1)
        morph_folder = os.path.join(self.morph_path, folder_name)
        morph_folder = self.normalize_path(morph_folder)
        print(f"Looking for morph images in: {morph_folder}")

        if os.path.exists(morph_folder):
            for img_file in os.listdir(morph_folder):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(morph_folder, img_file)
                    img_path = self.normalize_path(img_path)  # Normalize path
                    all_images.append(img_path)
                    all_labels.append(1)  # Morph label

                    # Add placeholder paths for landmarks and face labels
                    all_path_lm.append(None)
                    all_face_labels.append(None)

        # If train/val phase, split the dataset
        if self.phase != 'test':
            # Combine data and shuffle with fixed seed for reproducibility
            combined = list(zip(all_images, all_labels, all_path_lm, all_face_labels))
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(combined)
            all_images, all_labels, all_path_lm, all_face_labels = zip(*combined)

            # Convert back to lists
            all_images = list(all_images)
            all_labels = list(all_labels)
            all_path_lm = list(all_path_lm)
            all_face_labels = list(all_face_labels)

            # Split according to train_val_split
            split_idx = int(len(all_images) * self.train_val_split)

            if self.phase == 'train':
                self.image_list = all_images[:split_idx]
                self.labels = all_labels[:split_idx]
                self.path_lm = all_path_lm[:split_idx]
                self.face_labels = all_face_labels[:split_idx]
            elif self.phase == 'val':
                self.image_list = all_images[split_idx:]
                self.labels = all_labels[split_idx:]
                self.path_lm = all_path_lm[split_idx:]
                self.face_labels = all_face_labels[split_idx:]
        else:
            # For test phase, use all images
            self.image_list = all_images
            self.labels = all_labels
            self.path_lm = all_path_lm
            self.face_labels = all_face_labels

    def save_to_csv(self, csv_path):
        """Save dataset to CSV file"""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Normalize all paths before saving to CSV
        normalized_image_list = [self.normalize_path(path) for path in self.image_list]

        # Create DataFrame
        df = pd.DataFrame({
            'image_path': normalized_image_list,
            'label': self.labels,
            'split': [self.phase] * len(self.image_list)
        })

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            try:
                # Load existing CSV and append new data
                existing_df = pd.read_csv(csv_path)

                # Normalize paths in existing DataFrame
                if 'image_path' in existing_df.columns:
                    existing_df['image_path'] = existing_df['image_path'].apply(
                        lambda path: self.normalize_path(path) if pd.notna(path) else path
                    )

                # Remove existing entries with the same split
                existing_df = existing_df[existing_df['split'] != self.phase]

                # Append new data
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing CSV file: {e}")
                print(f"Creating new CSV file at {csv_path}")
                # Continue with creating a new CSV file

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved at {csv_path} with {len(df)} entries")

    def load_from_csv(self, csv_path):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(csv_path)

            # Filter by split (train/val)
            df = df[df['split'] == self.phase]

            # Check if we have any data for this split
            if len(df) == 0:
                print(f"Warning: No data found for split '{self.phase}' in {csv_path}")
                print("Creating new dataset from folder structure")
                self.create_dataset()
                self.save_to_csv(csv_path)
                return

            # Normalize paths in the DataFrame
            if 'image_path' in df.columns:
                df['image_path'] = df['image_path'].apply(
                    lambda path: self.normalize_path(path) if pd.notna(path) else path
                )

            # Extract data
            self.image_list = df['image_path'].tolist()
            self.labels = df['label'].tolist()

            # Add placeholder paths for landmarks and face labels
            self.path_lm = [None] * len(self.image_list)
            self.face_labels = [None] * len(self.image_list)

            # Verify that the paths exist
            valid_paths = 0
            for path in self.image_list:
                if os.path.exists(path):
                    valid_paths += 1
                else:
                    print(f"Warning: Path does not exist: {path}")

            print(f"Loaded {len(self.image_list)} samples for '{self.phase}' split from {csv_path}")
            print(f"Found {valid_paths}/{len(self.image_list)} valid paths")

            if valid_paths < len(self.image_list) * 0.5:  # If less than 50% of paths are valid
                print("Warning: Less than 50% of paths are valid. Recreating dataset from folder structure.")
                self.create_dataset()
                self.save_to_csv(csv_path)

        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            print("Creating new dataset from folder structure")
            self.create_dataset()
            self.save_to_csv(csv_path)

    def normalize_path(self, path):
        """Normalize path to use forward slashes and make it absolute if possible"""
        if path is None:
            return None

        # Convert to absolute path if it's relative
        if not os.path.isabs(path):
            # Try different base directories
            possible_bases = [
                os.getcwd(),  # Current working directory
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
                "/cluster/home/aminurrs/SelfMAD_Combined"  # Cloud-specific path
            ]

            for base in possible_bases:
                abs_path = os.path.abspath(os.path.join(base, path))
                if os.path.exists(abs_path):
                    path = abs_path
                    break

        # Normalize path and convert backslashes to forward slashes
        return os.path.normpath(path).replace('\\', '/')

    def __getitem__(self, idx):
        """Get item with on-the-fly landmark and face label generation"""
        try:
            # Load image
            img_path = self.image_list[idx]

            # Normalize path using our comprehensive function
            img_path = self.normalize_path(img_path)

            # Try multiple path variations to find the file
            possible_paths = [
                img_path,  # Try the normalized path first
                os.path.join(".", img_path),  # Try relative to current directory
                os.path.basename(img_path),  # Try just the filename in current directory
                os.path.join("datasets", "morph", self.dataset_name, "train", os.path.basename(img_path)),  # Try standard location
                os.path.join("datasets", "bonafide", self.dataset_name, "train", os.path.basename(img_path)),  # Try standard location
                os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "morph", self.dataset_name, "train", os.path.basename(img_path)),  # Try cloud location
                os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "bonafide", self.dataset_name, "train", os.path.basename(img_path))  # Try cloud location
            ]

            # Try each path until we find one that exists
            img = None
            for path in possible_paths:
                path = self.normalize_path(path)
                if os.path.exists(path):
                    try:
                        img = np.array(Image.open(path))
                        # If we successfully loaded the image, update the path in our list for future use
                        if path != img_path:
                            print(f"Found image at alternative path: {path}")
                            self.image_list[idx] = path
                        break
                    except Exception as e:
                        print(f"Error loading image {path}: {str(e)}")
                        continue

            # If we couldn't find the image, create a blank one
            if img is None:
                print(f"Warning: Image file not found for index {idx}. Tried paths: {possible_paths}")
                # Create a blank image as a fallback
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

            # Generate landmarks and face labels on-the-fly
            try:
                if self.path_lm[idx] is None or self.face_labels[idx] is None:
                    # Use parent class methods to generate these
                    landmark = self.generate_landmarks(img)
                    face_label = self.generate_face_labels(img)
                else:
                    # Use existing landmarks and face labels
                    landmark = np.load(self.path_lm[idx])
                    face_label = np.load(self.face_labels[idx])
            except Exception as e:
                print(f"Error generating landmarks/face labels: {str(e)}")
                # Create fallback landmarks and face labels
                landmark = self.generate_landmarks(img)
                face_label = self.generate_face_labels(img)

            # Apply transformations using parent class methods
            if self.phase == 'train':
                try:
                    # Apply self-morphing or self-blending
                    if np.random.rand() < 0.5:
                        img_r, img_f, _ = self.self_blending(img.copy(), landmark.copy())
                    else:
                        img_r, img_f, _, _ = self.self_morphing(img.copy(), face_label.copy(), landmark.copy())

                    # Apply additional transformations
                    transformed = self.transforms(image=img_f.astype('uint8'), image1=img_r.astype('uint8'))
                    img_f = transformed['image']
                    img_r = transformed['image1']

                    # Apply frequency transform occasionally
                    if np.random.rand() < 0.1:
                        freq_transform = self.create_frequency_noise_transform(weight=np.random.uniform(0.025, 0.1))
                        img_f = freq_transform(image=img_f)['image']
                except Exception as e:
                    print(f"Error in transformations: {str(e)}")
                    # Create blank images as fallback
                    img_f = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                    img_r = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

                # Ensure both images have the same dimensions by resizing
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)  # Create a tuple of (width, height)
                img_f = cv2.resize(img_f, target_size, interpolation=cv2.INTER_LANCZOS4)
                img_r = cv2.resize(img_r, target_size, interpolation=cv2.INTER_LANCZOS4)

                # Return tuple of (img_f, img_r) as expected by collate_fn
                return img_f, img_r
            else:
                # For validation and testing, resize using Lanczos resampling
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)  # Create a tuple of (width, height)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # For validation, we need to create a fake img_f and img_r pair
                # We'll use the same image for both to maintain compatibility with the collate_fn
                return img, img
        except Exception as e:
            # Log the error with detailed information
            error_msg = f"Error in MorphDataset.__getitem__ for index {idx}, image path: {self.image_list[idx]}"
            logger.error(f"{error_msg}: {str(e)}")

            # Create blank images as fallback, but with a visible pattern to make it obvious
            if self.phase == 'train':
                # Create checkerboard pattern to make it obvious this is a fallback image
                img_f = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                img_r = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

                # Add a red X pattern to make it obvious this is an error image
                cv2.line(img_f, (0, 0), (self.image_size[0], self.image_size[1]), (0, 0, 255), 5)
                cv2.line(img_f, (self.image_size[0], 0), (0, self.image_size[1]), (0, 0, 255), 5)
                cv2.line(img_r, (0, 0), (self.image_size[0], self.image_size[1]), (0, 0, 255), 5)
                cv2.line(img_r, (self.image_size[0], 0), (0, self.image_size[1]), (0, 0, 255), 5)

                # Log that we're returning fallback images
                logger.warning(f"Returning fallback images for index {idx}")
                return img_f, img_r
            else:
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
                # Add a red X pattern to make it obvious this is an error image
                cv2.line(img, (0, 0), (self.image_size[0], self.image_size[1]), (0, 0, 255), 5)
                cv2.line(img, (self.image_size[0], 0), (0, self.image_size[1]), (0, 0, 255), 5)

                # Log that we're returning fallback images
                logger.warning(f"Returning fallback images for index {idx}")
                return img, img

    def generate_landmarks(self, img):
        """Generate facial landmarks for an image"""
        # Placeholder implementation - in a real scenario, you would use a landmark detector
        # For now, we'll create dummy landmarks based on image size
        h, w = img.shape[:2]

        # Create a more detailed face outline with more points for better convex hull
        # These points form a rough oval around the face
        num_points = 81
        landmarks = np.zeros((num_points, 2), dtype=np.int32)  # Use int32 for OpenCV compatibility

        # Create points in an oval shape
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = int(w/2 + (w/3) * np.cos(angle))
            y = int(h/2 + (h/3) * np.sin(angle))
            landmarks[i] = [x, y]

        # Add some key facial features
        # Eyes
        landmarks[0] = [int(w * 0.3), int(h * 0.3)]  # Left eye
        landmarks[1] = [int(w * 0.7), int(h * 0.3)]  # Right eye
        # Nose
        landmarks[2] = [int(w * 0.5), int(h * 0.5)]  # Nose
        # Mouth
        landmarks[3] = [int(w * 0.3), int(h * 0.7)]  # Left mouth
        landmarks[4] = [int(w * 0.7), int(h * 0.7)]  # Right mouth

        # Keep landmarks as int32 for OpenCV compatibility

        return landmarks

    def generate_face_labels(self, img):
        """Generate face parsing labels for an image"""
        # Placeholder implementation - in a real scenario, you would use a face parser
        # For now, we'll create a dummy face label map
        h, w = img.shape[:2]
        face_label = np.zeros((h, w), dtype=np.int32)

        # Create regions for different face parts (simplified)
        face_label[int(h*0.2):int(h*0.8), int(w*0.2):int(w*0.8)] = 1  # Face
        face_label[int(h*0.3):int(h*0.4), int(w*0.3):int(w*0.4)] = 2  # Left eye
        face_label[int(h*0.3):int(h*0.4), int(w*0.6):int(w*0.7)] = 3  # Right eye
        face_label[int(h*0.5):int(h*0.6), int(w*0.4):int(w*0.6)] = 4  # Mouth

        return face_label

    def save_to_csv(self, csv_path):
        """Save dataset to CSV file"""
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Normalize all paths before saving to CSV
        normalized_image_list = [self.normalize_path(path) for path in self.image_list]

        # Create DataFrame
        df = pd.DataFrame({
            'image_path': normalized_image_list,
            'label': self.labels,
            'split': [self.phase] * len(self.image_list)
        })

        # Check if the CSV file already exists
        if os.path.exists(csv_path):
            try:
                # Load existing CSV and append new data
                existing_df = pd.read_csv(csv_path)

                # Normalize paths in existing DataFrame
                if 'image_path' in existing_df.columns:
                    existing_df['image_path'] = existing_df['image_path'].apply(
                        lambda path: self.normalize_path(path) if pd.notna(path) else path
                    )

                # Remove existing entries with the same split
                existing_df = existing_df[existing_df['split'] != self.phase]

                # Append new data
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing CSV file: {e}")
                print(f"Creating new CSV file at {csv_path}")
                # Continue with creating a new CSV file

        # Save to CSV
        df.to_csv(csv_path, index=False)
        print(f"CSV file saved at {csv_path} with {len(df)} entries")

    def load_from_csv(self, csv_path):
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(csv_path)

            # Filter by split (train/val)
            df = df[df['split'] == self.phase]

            # Check if we have any data for this split
            if len(df) == 0:
                print(f"Warning: No data found for split '{self.phase}' in {csv_path}")
                print("Creating new dataset from folder structure")
                self.create_dataset()
                self.save_to_csv(csv_path)
                return

            # Normalize paths in the DataFrame
            if 'image_path' in df.columns:
                df['image_path'] = df['image_path'].apply(
                    lambda path: self.normalize_path(path) if pd.notna(path) else path
                )

            # Extract data
            self.image_list = df['image_path'].tolist()
            self.labels = df['label'].tolist()

            # Add placeholder paths for landmarks and face labels
            self.path_lm = [None] * len(self.image_list)
            self.face_labels = [None] * len(self.image_list)

            # Verify that the paths exist
            valid_paths = 0
            for path in self.image_list:
                if os.path.exists(path):
                    valid_paths += 1
                else:
                    print(f"Warning: Path does not exist: {path}")

            print(f"Loaded {len(self.image_list)} samples for '{self.phase}' split from {csv_path}")
            print(f"Found {valid_paths}/{len(self.image_list)} valid paths")

            if valid_paths < len(self.image_list) * 0.5:  # If less than 50% of paths are valid
                print("Warning: Less than 50% of paths are valid. Recreating dataset from folder structure.")
                self.create_dataset()
                self.save_to_csv(csv_path)

        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            print("Creating new dataset from folder structure")
            self.create_dataset()
            self.save_to_csv(csv_path)

    def __len__(self):
        return len(self.image_list)


class CombinedMorphDataset(selfMAD_Dataset):
    """Dataset for combining multiple morph detection datasets for training"""
    def __init__(self, dataset_names, phase='train', image_size=224, train_val_split=0.8, csv_path=None):
        """
        Custom dataset adapter for combining multiple morph detection datasets

        Args:
            dataset_names: List of dataset names to combine (e.g., ["LMA", "MIPGAN_I"])
            phase: 'train', 'val', or 'test'
            image_size: Size of the images
            train_val_split: Ratio of training data (0.8 = 80% train, 20% val)
            csv_path: Path to save/load CSV file with image paths and labels
        """
        # Initialize transforms from parent class first
        super().__init__(phase=phase, image_size=image_size, datapath=None)

        self.dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        self.train_val_split = train_val_split

        # Ensure image_size is properly handled
        if isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)

        # Initialize lists for images and labels
        self.image_list = []
        self.labels = []
        self.path_lm = []  # For landmarks
        self.face_labels = []  # For face labels

        # CSV handling
        self.csv_path = csv_path
        if csv_path is not None and os.path.exists(csv_path) and phase != 'test':
            # Load from existing CSV
            self.load_from_csv(csv_path)
        else:
            # Create new dataset
            self.create_dataset()
            if csv_path is not None and phase != 'test':
                self.save_to_csv(csv_path)

    def create_dataset(self):
        """Create dataset by combining multiple datasets"""
        if self.phase == 'test':
            # For test phase, use the 'test' folder
            folder_name = 'test'
        else:
            # For train/val phase, use the 'train' folder
            folder_name = 'train'

        # Load all images first, then split into train/val
        all_images = []
        all_labels = []
        all_path_lm = []
        all_face_labels = []

        # Try multiple possible locations for datasets
        possible_base_dirs = [
            os.environ.get("DATASETS_DIR"),  # First check environment variable
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"),  # Check relative to script
            os.path.join(".", "datasets"),  # Check in current directory
            os.path.join("..", "datasets"),  # Check one level up
            "/cluster/home/aminurrs/SelfMAD_Combined/datasets",  # Check specific cloud path
            os.path.abspath("datasets")  # Check absolute path
        ]

        # Use the first directory that exists, or default to the second option
        project_root_guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                        os.path.join(project_root_guess, "datasets"))

        print(f"CombinedMorphDataset using datasets directory: {base_dir}")

        # Process each dataset
        for dataset_name in self.dataset_names:
            print(f"Loading dataset: {dataset_name}")

            # Load bonafide images (label 0)
            bonafide_folder = os.path.join(base_dir, "bonafide", dataset_name, folder_name)
            bonafide_folder = self.normalize_path(bonafide_folder)
            print(f"Looking for bonafide images in: {bonafide_folder}")

            if os.path.exists(bonafide_folder):
                for img_file in os.listdir(bonafide_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(bonafide_folder, img_file)
                        img_path = self.normalize_path(img_path)  # Normalize path
                        all_images.append(img_path)
                        all_labels.append(0)  # Bonafide label

                        # Add placeholder paths for landmarks and face labels
                        all_path_lm.append(None)
                        all_face_labels.append(None)

            # Load morph images (label 1)
            morph_folder = os.path.join(base_dir, "morph", dataset_name, folder_name)
            morph_folder = self.normalize_path(morph_folder)
            print(f"Looking for morph images in: {morph_folder}")

            if os.path.exists(morph_folder):
                for img_file in os.listdir(morph_folder):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(morph_folder, img_file)
                        img_path = self.normalize_path(img_path)  # Normalize path
                        all_images.append(img_path)
                        all_labels.append(1)  # Morph label

                        # Add placeholder paths for landmarks and face labels
                        all_path_lm.append(None)
                        all_face_labels.append(None)

        print(f"Total images loaded from all datasets: {len(all_images)}")
        print(f"Bonafide images: {all_labels.count(0)}")
        print(f"Morph images: {all_labels.count(1)}")

        # If train/val phase, split the dataset
        if self.phase != 'test':
            # Combine data and shuffle with fixed seed for reproducibility
            combined = list(zip(all_images, all_labels, all_path_lm, all_face_labels))
            random.seed(42)  # Fixed seed for reproducibility
            random.shuffle(combined)
            all_images, all_labels, all_path_lm, all_face_labels = zip(*combined)

            # Convert back to lists
            all_images = list(all_images)
            all_labels = list(all_labels)
            all_path_lm = list(all_path_lm)
            all_face_labels = list(all_face_labels)

            # Split according to train_val_split
            split_idx = int(len(all_images) * self.train_val_split)

            if self.phase == 'train':
                self.image_list = all_images[:split_idx]
                self.labels = all_labels[:split_idx]
                self.path_lm = all_path_lm[:split_idx]
                self.face_labels = all_face_labels[:split_idx]
            elif self.phase == 'val':
                self.image_list = all_images[split_idx:]
                self.labels = all_labels[split_idx:]
                self.path_lm = all_path_lm[split_idx:]
                self.face_labels = all_face_labels[split_idx:]
        else:
            # For test phase, use all images
            self.image_list = all_images
            self.labels = all_labels
            self.path_lm = all_path_lm
            self.face_labels = all_face_labels

        print(f"Dataset '{self.phase}' created with {len(self.image_list)} images")


class TestMorphDataset(Dataset):
    """Dataset for testing morph detection models"""
    def __init__(self, dataset_name, image_size=224):
        self.dataset_name = dataset_name

        # Ensure image_size is properly handled
        if isinstance(image_size, tuple):
            self.image_size = image_size
        else:
            self.image_size = (image_size, image_size)

        # Initialize lists
        self.image_paths = []
        self.labels = []

        # Try multiple possible locations for datasets
        possible_base_dirs = [
            os.environ.get("DATASETS_DIR"),  # First check environment variable
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"),  # Check relative to script
            os.path.join(".", "datasets"),  # Check in current directory
            os.path.join("..", "datasets"),  # Check one level up
            "/cluster/home/aminurrs/SelfMAD_Combined/datasets",  # Check specific cloud path
            os.path.abspath("datasets")  # Check absolute path
        ]

        # Use the first directory that exists, or default to the second option
        project_root_guess = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                        os.path.join(project_root_guess, "datasets"))

        print(f"TestMorphDataset using datasets directory: {base_dir}")

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

    def __len__(self):
        return len(self.image_paths)

    def normalize_path(self, path):
        """Normalize path to use forward slashes and make it absolute if possible"""
        if path is None:
            return None

        # Convert to absolute path if it's relative
        if not os.path.isabs(path):
            # Try different base directories
            possible_bases = [
                os.getcwd(),  # Current working directory
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Project root
                "/cluster/home/aminurrs/SelfMAD_Combined"  # Cloud-specific path
            ]

            for base in possible_bases:
                abs_path = os.path.abspath(os.path.join(base, path))
                if os.path.exists(abs_path):
                    path = abs_path
                    break

        # Normalize path and convert backslashes to forward slashes
        return os.path.normpath(path).replace('\\', '/')

    def __getitem__(self, idx):
        try:
            # Load and preprocess image
            img_path = self.image_paths[idx]

            # Normalize path using our comprehensive function
            img_path = self.normalize_path(img_path)

            # Try multiple path variations to find the file
            possible_paths = [
                img_path,  # Try the normalized path first
                os.path.join(".", img_path),  # Try relative to current directory
                os.path.basename(img_path),  # Try just the filename in current directory
                os.path.join("datasets", "morph", self.dataset_name, "test", os.path.basename(img_path)),  # Try standard location
                os.path.join("datasets", "bonafide", self.dataset_name, "test", os.path.basename(img_path)),  # Try standard location
                os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "morph", self.dataset_name, "test", os.path.basename(img_path)),  # Try cloud location
                os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "bonafide", self.dataset_name, "test", os.path.basename(img_path))  # Try cloud location
            ]

            # Try each path until we find one that exists
            img = None
            for path in possible_paths:
                path = self.normalize_path(path)
                if os.path.exists(path):
                    try:
                        img = np.array(Image.open(path))
                        # If we successfully loaded the image, update the path in our list for future use
                        if path != img_path:
                            print(f"Found image at alternative path: {path}")
                            self.image_paths[idx] = path
                        break
                    except Exception as e:
                        print(f"Error loading image {path}: {str(e)}")
                        continue

            # If we couldn't find the image, create a blank one
            if img is None:
                print(f"Warning: Image file not found for index {idx}. Tried paths: {possible_paths}")
                # Create a blank image as a fallback
                img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

            # Resize to expected dimensions using Lanczos resampling
            img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LANCZOS4)

            # Convert to PyTorch format
            img = img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Return in the format expected by the evaluate function
            return {'img': img, 'label': torch.tensor(self.labels[idx])}
        except Exception as e:
            # Log the error with detailed information
            error_msg = f"Error in TestMorphDataset.__getitem__ for index {idx}, image path: {self.image_paths[idx]}"
            logger.error(f"{error_msg}: {str(e)}")

            # Create a blank image as a fallback, but with a visible pattern
            img = torch.zeros((3, self.image_size[0], self.image_size[1]), dtype=torch.float32)

            # Add a red X pattern to make it obvious this is an error image (in tensor format)
            # Red channel (index 0)
            for i in range(min(self.image_size[0], self.image_size[1])):
                if i < self.image_size[0] and i < self.image_size[1]:
                    img[0, i, i] = 1.0  # Diagonal from top-left to bottom-right
                    img[0, i, self.image_size[1]-1-i] = 1.0  # Diagonal from top-right to bottom-left

            # Log that we're returning a fallback image
            logger.warning(f"Returning fallback image for index {idx}")
            return {'img': img, 'label': torch.tensor(self.labels[idx])}
