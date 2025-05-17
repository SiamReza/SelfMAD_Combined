import argparse
import torch
from utils.model import Detector
from PIL import Image
import cv2
import numpy as np
import os

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MODEL
    model_state_path = args.model_path
    model = Detector(model=args.model_type)
    # Set weights_only=False to allow loading NumPy data types in PyTorch 2.6+
    model_state = torch.load(model_state_path, weights_only=False)
    model.load_state_dict(model_state['model'])
    model.train(mode=False)
    model.to(device)

    # IMG
    if args.model_type == "vit_mae_large":
        image_size = 224  # ViT-MAE uses 224x224 images
    else:
        image_size = 384 if "hrnet" in args.model_type else 380

    # Normalize path (replace backslashes with forward slashes)
    input_path = os.path.normpath(args.input_path).replace('\\', '/')

    try:
        # Check if file exists
        if os.path.exists(input_path):
            try:
                img = np.array(Image.open(input_path))
            except Exception as e:
                print(f"Error loading image {input_path}: {str(e)}")
                # Create a blank image as a fallback
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        else:
            # Try relative to current directory
            alt_path = os.path.join(".", input_path)
            alt_path = os.path.normpath(alt_path).replace('\\', '/')
            if os.path.exists(alt_path):
                try:
                    img = np.array(Image.open(alt_path))
                except Exception as e:
                    print(f"Error loading image {alt_path}: {str(e)}")
                    # Create a blank image as a fallback
                    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                print(f"Warning: Image file not found: {input_path} or {alt_path}")
                # Create a blank image as a fallback
                img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        # Process the image
        img = cv2.resize(img, (image_size, image_size))
        img = img.transpose((2,0,1))
        img = img.astype('float32')/255
        img = img[np.newaxis, ...]
        img = torch.from_numpy(img).to(device, non_blocking=True).float()

        with torch.no_grad():
            output = model(img).softmax(1)[:, 1].cpu().data.numpy()[0]
        print("Confidence:", output)
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        print("Failed to process the image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('-m', dest='model_type', type=str, required=True, help='Type of the model, e.g. hrnet_w18')
    parser.add_argument('-in', dest='input_path', type=str, required=True, help='Path to input image')
    parser.add_argument('-p', dest='model_path', type=str, required=True, help='Path to saved model')

    args = parser.parse_args()
    main(args)