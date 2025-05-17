from torch.utils.data import Dataset, ConcatDataset
import os
import numpy as np
from PIL import Image
import cv2
import random
import pandas as pd
from utils.selfMAD import selfMAD_Dataset
import torch
import datetime


class OriginalMorphDataset(Dataset):
    def __init__(self, datapath, image_size, phase='train', transform=None):
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

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img=np.array(Image.open(self.image_paths[idx]))
        label=self.labels[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))
        img=img.transpose((2,0,1))
        img = img.astype('float32')/255
        if self.transform:
            img = self.transform(img)
        return img, label


class MorphDataset(Dataset):
    def __init__(self, dataset_name, phase='train', image_size=384, train_val_split=0.8, csv_path=None):
        """
        Custom dataset adapter for morph detection

        Args:
            dataset_name: Name of the dataset (LMA, LMA_UBO, etc.)
            phase: 'train', 'val', or 'test'
            image_size: Size of the images
            train_val_split: Ratio of training data (0.8 = 80% train, 20% val)
            csv_path: Path to save/load CSV file with image paths and labels
        """
        self.phase = phase
        self.image_size = image_size

        self.dataset_name = dataset_name
        self.train_val_split = train_val_split

        # Base paths - use os.path.join for cross-platform compatibility
        # Get the base directory from environment or use a relative path
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
        base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))

        print(f"Using datasets directory: {base_dir}")

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

    def create_dataset(self):
        """Create dataset from folder structure"""
        # Determine which folder to use based on phase
        folder_name = "train" if self.phase != 'test' else "test"

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

    def __getitem__(self, idx):
        """Get item from the dataset"""
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
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

            # For validation and testing, resize and return as dictionary
            if self.phase != 'train':
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # Convert to PyTorch format
                img = img.astype('float32') / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)

                # Return dictionary with image and label
                return {'img': img, 'label': self.labels[idx]}
            else:
                # For training, create a fake image
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)

                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # Create a fake image by applying simple transformations
                fake_img = img.copy()

                # Apply some basic transformations to create a fake image
                # Adjust brightness
                fake_img = cv2.convertScaleAbs(fake_img, alpha=1.1, beta=10)

                # Convert to PyTorch format
                img = img.astype('float32') / 255.0
                fake_img = fake_img.astype('float32') / 255.0

                # Return tuple of (fake_img, real_img) as expected by collate_fn
                return fake_img, img
        except Exception as e:
            print(f"Unexpected error in MorphDataset.__getitem__ for index {idx}, path: {self.image_list[idx]}: {str(e)}")
            # Create blank images as fallback
            if isinstance(self.image_size, tuple):
                target_size = self.image_size
            else:
                target_size = (self.image_size, self.image_size)

            if self.phase != 'train':
                # For validation/testing, return a dictionary with blank image
                img = np.zeros((3, target_size[1], target_size[0]), dtype=np.float32)
                return {'img': torch.tensor(img), 'label': self.labels[idx]}
            else:
                # For training, return a tuple of blank images
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
                fake_img = img.copy()
                return fake_img, img

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

    def collate_fn(self, batch):
        """Custom collate function for the dataloader"""
        if self.phase == 'train':
            # For training, batch contains tuples of (fake_img, real_img)
            fake_imgs = []
            real_imgs = []
            for fake_img, real_img in batch:
                fake_imgs.append(torch.from_numpy(fake_img.transpose((2, 0, 1))))
                real_imgs.append(torch.from_numpy(real_img.transpose((2, 0, 1))))

            # Stack images and create labels
            fake_imgs = torch.stack(fake_imgs)
            real_imgs = torch.stack(real_imgs)
            fake_labels = torch.ones(len(batch), dtype=torch.long)
            real_labels = torch.zeros(len(batch), dtype=torch.long)

            # Combine fake and real images
            imgs = torch.cat([fake_imgs, real_imgs], dim=0)
            labels = torch.cat([fake_labels, real_labels], dim=0)

            return {'img': imgs, 'label': labels}
        else:
            # For validation and testing, batch contains dictionaries
            imgs = []
            labels = []
            for item in batch:
                imgs.append(item['img'])
                labels.append(item['label'])

            return {'img': torch.stack(imgs), 'label': torch.tensor(labels, dtype=torch.long)}

    def worker_init_fn(self, worker_id):
        """Initialize worker for dataloader"""
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_list)


class CombinedMorphDataset(Dataset):
    """Dataset for combining multiple morph detection datasets for training"""
    def __init__(self, dataset_names, phase='train', image_size=384, train_val_split=0.8, csv_path=None):
        """
        Custom dataset adapter for combining multiple morph detection datasets

        Args:
            dataset_names: List of dataset names to combine (e.g., ["LMA", "MIPGAN_I"])
            phase: 'train', 'val', or 'test'
            image_size: Size of the images
            train_val_split: Ratio of training data (0.8 = 80% train, 20% val)
            csv_path: Path to save/load CSV file with image paths and labels
        """
        self.phase = phase
        self.image_size = image_size

        self.dataset_names = dataset_names if isinstance(dataset_names, list) else [dataset_names]
        self.train_val_split = train_val_split

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

    def create_dataset(self):
        """Create dataset by combining multiple datasets"""
        # Determine which folder to use based on phase
        folder_name = "train" if self.phase != 'test' else "test"

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
        base_dir = next((d for d in possible_base_dirs if d and os.path.exists(d)),
                        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets"))

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

    def __getitem__(self, idx):
        """Get item from the dataset"""
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
            ]

            # Add paths for each dataset
            for dataset_name in self.dataset_names:
                possible_paths.extend([
                    os.path.join("datasets", "morph", dataset_name, "train", os.path.basename(img_path)),  # Try standard location
                    os.path.join("datasets", "bonafide", dataset_name, "train", os.path.basename(img_path)),  # Try standard location
                    os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "morph", dataset_name, "train", os.path.basename(img_path)),  # Try cloud location
                    os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", "bonafide", dataset_name, "train", os.path.basename(img_path))  # Try cloud location
                ])

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
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

            # For validation and testing, resize and return as dictionary
            if self.phase != 'train':
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # Convert to PyTorch format
                img = img.astype('float32') / 255.0
                img = torch.from_numpy(img).permute(2, 0, 1)

                # Return dictionary with image and label
                return {'img': img, 'label': self.labels[idx]}
            else:
                # For training, create a fake image
                if isinstance(self.image_size, tuple):
                    target_size = self.image_size
                else:
                    target_size = (self.image_size, self.image_size)

                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

                # Create a fake image by applying simple transformations
                fake_img = img.copy()

                # Apply some basic transformations to create a fake image
                # Adjust brightness
                fake_img = cv2.convertScaleAbs(fake_img, alpha=1.1, beta=10)

                # Convert to PyTorch format
                img = img.astype('float32') / 255.0
                fake_img = fake_img.astype('float32') / 255.0

                # Return tuple of (fake_img, real_img) as expected by collate_fn
                return fake_img, img
        except Exception as e:
            print(f"Unexpected error in CombinedMorphDataset.__getitem__ for index {idx}, path: {self.image_list[idx]}: {str(e)}")
            # Create blank images as fallback
            if isinstance(self.image_size, tuple):
                target_size = self.image_size
            else:
                target_size = (self.image_size, self.image_size)

            if self.phase != 'train':
                # For validation/testing, return a dictionary with blank image
                img = np.zeros((3, target_size[1], target_size[0]), dtype=np.float32)
                return {'img': torch.tensor(img), 'label': self.labels[idx]}
            else:
                # For training, return a tuple of blank images
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
                fake_img = img.copy()
                return fake_img, img

    def collate_fn(self, batch):
        """Custom collate function for the dataloader"""
        if self.phase == 'train':
            # For training, batch contains tuples of (fake_img, real_img)
            fake_imgs = []
            real_imgs = []
            for fake_img, real_img in batch:
                fake_imgs.append(torch.from_numpy(fake_img.transpose((2, 0, 1))))
                real_imgs.append(torch.from_numpy(real_img.transpose((2, 0, 1))))

            # Stack images and create labels
            fake_imgs = torch.stack(fake_imgs)
            real_imgs = torch.stack(real_imgs)
            fake_labels = torch.ones(len(batch), dtype=torch.long)
            real_labels = torch.zeros(len(batch), dtype=torch.long)

            # Combine fake and real images
            imgs = torch.cat([fake_imgs, real_imgs], dim=0)
            labels = torch.cat([fake_labels, real_labels], dim=0)

            return {'img': imgs, 'label': labels}
        else:
            # For validation and testing, batch contains dictionaries
            imgs = []
            labels = []
            for item in batch:
                imgs.append(item['img'])
                labels.append(item['label'])

            return {'img': torch.stack(imgs), 'label': torch.tensor(labels, dtype=torch.long)}

    def worker_init_fn(self, worker_id):
        """Initialize worker for dataloader"""
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def __len__(self):
        """Return the length of the dataset"""
        return len(self.image_list)


class PartialMorphDataset(Dataset):
    def __init__(self, datapath, image_size, transform=None, method=None):
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
                os.path.join("datasets", self.dataset_name, os.path.basename(img_path)),  # Try standard location
                os.path.join("/cluster/home/aminurrs/SelfMAD_Combined/datasets", self.dataset_name, os.path.basename(img_path))  # Try cloud location
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
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            # Process the image
            label = self.labels[idx]
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img.transpose((2,0,1))
            img = img.astype('float32')/255
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Unexpected error in PartialMorphDataset.__getitem__ for index {idx}: {str(e)}")
            # Create a blank image as a fallback
            img = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
            return torch.tensor(img), self.labels[idx]


class TestMorphDataset(Dataset):
    """Dataset for testing morph detection models"""
    def __init__(self, dataset_name, image_size=384):
        self.dataset_name = dataset_name
        self.image_size = image_size

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

        # Similarly for output directory
        possible_output_dirs = [
            os.environ.get("OUTPUT_DIR"),  # First check environment variable
            os.path.join(project_root_guess, "output"),  # Check relative to script
            os.path.join(".", "output"),  # Check in current directory
            os.path.join("..", "output"),  # Check one level up
            "/cluster/home/aminurrs/SelfMAD_Combined/output",  # Check specific cloud path
            os.path.abspath("output")  # Check absolute path
        ]

        output_dir = next((d for d in possible_output_dirs if d and os.path.exists(d)),
                          os.path.join(project_root_guess, "output"))

        print(f"TestMorphDataset using output directory: {output_dir}")

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
            log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "missing_test_data.log")
            with open(log_file, "a") as f:
                f.write(f"[{datetime.datetime.now()}] No test images found for {dataset_name}. Paths checked: {bonafide_path}, {morph_path}\n")

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
                img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

            # Resize to expected dimensions using Lanczos resampling
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LANCZOS4)

            # Convert to PyTorch format
            img = img.astype('float32') / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)

            # Return in the format expected by the evaluate function
            return img, self.labels[idx]
        except Exception as e:
            print(f"Unexpected error in __getitem__ for index {idx}: {str(e)}")
            # Create a blank image as a fallback
            img = np.zeros((3, self.image_size, self.image_size), dtype=np.float32)
            return torch.tensor(img), self.labels[idx]

    def collate_fn(self, batch):
        """Custom collate function for the dataloader"""
        imgs = []
        labels = []
        for img, label in batch:
            imgs.append(img)
            labels.append(label)

        return {'img': torch.stack(imgs), 'label': torch.tensor(labels, dtype=torch.long)}

    def worker_init_fn(self, worker_id):
        """Initialize worker for dataloader"""
        np.random.seed(np.random.get_state()[1][0] + worker_id)

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

def default_datasets(image_size, datasets="original", config=None):
    """
    Create default datasets for evaluation.

    Args:
        image_size: Size of the images
        datasets: Type of datasets to create ("original" or "custom_morph")
        config: Configuration dictionary with dataset paths

    Returns:
        Dictionary of datasets
    """
    assert datasets in ["original", "custom_morph"]
    if datasets == "original":
        FRGC_datapath = config["FRGC_path"]
        FERET_datapath = config["FERET_path"]
        FRLL_datapath = config["FRLL_path"]

        test_datasets = {
            "FRGC": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRGC_datapath, method='stylegan')
            },
            "FERET": {
                "fm": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FERET_datapath, method='stylegan')
            },
            "FRLL": {
                "amsl": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='amsl'),
                "fm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='facemorpher'),
                "cv": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='opencv'),
                "sg": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='stylegan'),
                "wm": PartialMorphDataset(image_size=image_size, datapath=FRLL_datapath, method='webmorph')
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
                    "test": TestMorphDataset(dataset_name=dataset_name, image_size=image_size)
                }

        return test_datasets