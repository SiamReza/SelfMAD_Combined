# SelfMAD: Self-supervised Morphing Attack Detection

This repository contains the implementation of SelfMAD, a self-supervised learning approach for morphing attack detection.

> [**SelfMAD: Enhancing Generalization and Robustness in Morphing Attack Detection via Self-Supervised Learning**](https://arxiv.org/abs/2504.05504),
> Marija Ivanovska, Leon Todorov, Naser Damer, Deepak Kumar Jain, Peter Peer, Vitomir Štruc
> *FG 2025 Preprint*

## Repository Structure

The repository contains two main implementations:

1. **SelfMAD-main**: The original implementation supporting CNN-based architectures (EfficientNet, ResNet, HRNet, etc.)
2. **SelfMAD-siam**: An extended implementation that adds support for Vision Transformer models (ViT-MAE large)

## Key Features

- Self-supervised morphing techniques for data augmentation
- Support for multiple model architectures
- Cross-dataset evaluation capabilities
- Enhanced data augmentation pipeline

## Image Preprocessing

### SelfMAD-main
- Uses 384×384 pixels for HRNet models and 380×380 pixels for other models
- Applies face detection, cropping, and self-morphing/blending
- Uses basic augmentations (RGBShift, HueSaturationValue, etc.)
- Normalizes by dividing by 255

### SelfMAD-siam
- Uses 224×224 pixels for ViT-MAE large model
- Uses 384×384 pixels for HRNet models and 380×380 pixels for other models
- Applies enhanced augmentations
- Adds ImageNet normalization for ViT-MAE model

## Dataset Structure

The repository maintains the following dataset structure:

```
datasets/
├── bonafide/
│   ├── LMA/
│   ├── LMA_UBO/
│   ├── MIPGAN_I/
│   ├── MIPGAN_II/
│   ├── MorDiff/
│   └── StyleGAN/
└── morph/
    ├── LMA/
    ├── LMA_UBO/
    ├── MIPGAN_I/
    ├── MIPGAN_II/
    ├── MorDiff/
    └── StyleGAN/
```

Note: The actual image files are not included in this repository.

## Setup Instructions

To set up the code, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/SiamReza/SelfMAD_Combined.git
   cd SelfMAD_Combined
   ```

2. Download the ViT-MAE model:
   The ViT-MAE large model file is too large for GitHub. You need to download it separately:
   - Download the ViT-MAE large model from [Hugging Face](https://huggingface.co/facebook/vit-mae-large)
   - Save the `pytorch_model.bin` file to the `models/vit_mae/` directory

3. Create and activate the conda environment:
   ```
   # For SelfMAD-main
   cd SelfMAD-main/conda
   conda env create -f environment.yml
   conda activate selfmad

   # For SelfMAD-siam
   cd SelfMAD-siam/conda
   conda env create -f environment.yml
   conda activate selfmad-siam
   ```

4. Run the experiments:
   ```
   python run_experiments.py
   ```

## Documentation

The following documentation files are included:

- `SelfMAD-main/README.md`: Documentation for the original SelfMAD implementation
- `SelfMAD-siam/README.md`: Documentation for the extended SelfMAD implementation
- `SelfMAD-siam/selfMAD_ins.md`: Instructions for adapting SelfMAD for ViT-MAE
- `SelfMAD-siam/run_code.md`: Instructions for running the code

## Citation

If you use this code in your research, please cite:

```
@article{selfmad2025,
  title={SelfMAD: Enhancing Generalization and Robustness in Morphing Attack Detection via Self-Supervised Learning},
  author={Ivanovska, Marija and Todorov, Leon and Damer, Naser and Jain, Deepak Kumar and Peer, Peter and Štruc, Vitomir},
  journal={FG 2025 Preprint},
  year={2025}
}
```
