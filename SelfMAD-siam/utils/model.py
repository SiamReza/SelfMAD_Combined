import torch
from torch import nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import swin_v2_b, resnet152
from torchvision.models import Swin_V2_B_Weights, ResNet152_Weights
import timm
from utils.sam import SAM
import os
import json
import numpy as np
from transformers import ViTMAEModel
from torch.optim import AdamW
from utils.loss import smooth_targets, get_loss_function

class Detector(nn.Module):

    def __init__(self, model="hrnet_w18_multi", lr=5e-4):
        super(Detector, self).__init__()
        if model == "efficientnet-b4":
            self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        elif model == "efficientnet-b7":
            self.net=EfficientNet.from_pretrained("efficientnet-b7",advprop=True,num_classes=2)
        elif model == "swin":
            self.net=swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
            self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
        elif model == "resnet":
            self.net = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
            self.net.head = nn.Linear(in_features=1024, out_features=2, bias=True)
        elif model == "hrnet_w18":
            self.net = timm.create_model('hrnet_w18', pretrained=True, num_classes=2)
        elif model == "hrnet_w32":
            self.net = timm.create_model('hrnet_w32', pretrained=True, num_classes=2)
        elif model == "hrnet_w44":
            self.net = timm.create_model('hrnet_w44', pretrained=True, num_classes=2)
        elif model == "hrnet_w64":
            self.net = timm.create_model('hrnet_w64', pretrained=True, num_classes=2)
        elif model == "vit_mae_large":
            # Get the directory where the script is located
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # utils/ -> SelfMAD-siam/

            project_root = os.path.dirname(os.path.dirname(script_dir))

            # Load ViT-MAE model using absolute path for cross-platform compatibility
            vit_mae_path = os.path.join(project_root, "models", "vit_mae")

            # Print the path for debugging
            print(f"Loading ViT-MAE model from: {vit_mae_path}")

            # Check if the path exists
            if not os.path.exists(vit_mae_path):
                print(f"Warning: ViT-MAE model path does not exist: {vit_mae_path}")
                print("Falling back to Hugging Face model")
                # Fall back to Hugging Face model
                vit_mae_path = "facebook/vit-mae-large"

            # Load the pre-trained ViT-MAE model
            self.net = ViTMAEModel.from_pretrained(vit_mae_path)

            # Get hidden size from config
            if os.path.isdir(vit_mae_path) and os.path.exists(os.path.join(vit_mae_path, "config.json")):
                # Local model with config file
                with open(os.path.join(vit_mae_path, "config.json"), 'r') as f:
                    config = json.load(f)
                    hidden_size = config.get("hidden_size", 1024)
                    num_layers = config.get("num_hidden_layers", 24)
            else:
                # Hugging Face model or no config file
                # Use default values for ViT-MAE-large
                hidden_size = 1024
                num_layers = 24

            # Try to get configuration from automation_config.py if it exists
            try:
                import sys
                sys.path.append('..')
                from automation_config import get_config
                cfg = get_config()
                num_layers_to_unfreeze = cfg.get("siam_vit_unfreeze_layers", 4)
            except (ImportError, AttributeError):
                # Default value if config is not available
                num_layers_to_unfreeze = 4

            # First freeze all encoder parameters
            for param in self.net.parameters():
                param.requires_grad = False

            # Then unfreeze the specified number of transformer blocks
            if num_layers_to_unfreeze > 0:
                for i in range(num_layers - num_layers_to_unfreeze, num_layers):
                    for param in self.net.encoder.layer[i].parameters():
                        param.requires_grad = True

            # Add enhanced classification head with GELU and LayerNorm
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 768),
                nn.GELU(),
                nn.LayerNorm(768),
                nn.Dropout(0.2),
                nn.Linear(768, 384),
                nn.GELU(),
                nn.LayerNorm(384),
                nn.Dropout(0.1),
                nn.Linear(384, 2)
            )

        # Try to get configuration from automation_config.py if it exists
        try:
            import sys
            sys.path.append('..')
            from automation_config import get_config
            cfg = get_config()
            use_adamw = cfg.get("siam_vit_use_adamw", True)
            encoder_weight_decay = cfg.get("siam_vit_encoder_weight_decay", 0.05)
            classifier_weight_decay = cfg.get("siam_vit_classifier_weight_decay", 0.01)
            label_smoothing = cfg.get("siam_vit_label_smoothing", 0.1)
        except (ImportError, AttributeError):
            # Default values if config is not available
            use_adamw = True
            encoder_weight_decay = 0.05
            classifier_weight_decay = 0.01
            label_smoothing = 0.1

        # Get loss function configuration
        try:
            import sys
            sys.path.append('..')
            from automation_config import get_config
            cfg = get_config()
            loss_type = cfg.get("siam_loss_type", "bce_smoothing")
            label_smoothing = cfg.get("siam_vit_label_smoothing", 0.1)
            focal_alpha = cfg.get("siam_focal_alpha", 0.25)
            focal_gamma = cfg.get("siam_focal_gamma", 2.0)
        except (ImportError, AttributeError):
            # Default values if config is not available
            loss_type = "bce_smoothing"
            label_smoothing = 0.1
            focal_alpha = 0.25
            focal_gamma = 2.0

        # Create loss function based on configuration
        self.cel = get_loss_function(
            loss_type=loss_type,
            label_smoothing=label_smoothing,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma
        )

        # Use AdamW optimizer with layer-wise learning rates for ViT MAE
        if model == "vit_mae_large" and use_adamw:
            # Create parameter groups with different learning rates
            encoder_params = [p for n, p in self.named_parameters() if 'net.encoder' in n and p.requires_grad]
            classifier_params = [p for n, p in self.named_parameters() if 'classifier' in n]

            self.optimizer = AdamW([
                {'params': encoder_params, 'lr': lr * 0.1, 'weight_decay': encoder_weight_decay},  # Lower LR for encoder
                {'params': classifier_params, 'lr': lr, 'weight_decay': classifier_weight_decay}   # Higher LR for classifier
            ])
        else:
            # Use SAM optimizer for other models or if AdamW is disabled
            self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=lr, momentum=0.9)

    def forward(self, x):
        try:
            if hasattr(self, 'classifier'):
                # For ViT-MAE model
                # Always ensure the input is in the correct format for ViT-MAE
                # ViT-MAE expects [B, C, H, W] with C=3

                # Handle different tensor formats
                if x.dim() == 4:
                    if x.shape[1] != 3:  # Not in the expected channel dimension
                        if x.shape[3] == 3:  # In [B, H, W, C] format
                            x = x.permute(0, 3, 1, 2)
                        elif x.shape[2] == 3:  # In [B, H, C, W] format
                            x = x.permute(0, 2, 1, 3)
                elif x.dim() == 3:
                    if x.shape[0] == 3:  # Single image in [C, H, W] format
                        x = x.unsqueeze(0)  # Add batch dimension
                    elif x.shape[2] == 3:  # Single image in [H, W, C] format
                        x = x.permute(2, 0, 1).unsqueeze(0)  # Convert to [B, C, H, W]

                # Ensure pixel values are in the expected range [0, 1]
                if x.max() > 1.0:
                    x = x / 255.0

                # Final check and correction
                if x.dim() == 4 and x.shape[1] != 3:
                    # Last resort: try to reshape if possible
                    if x.shape[1] == 1 and x.shape[2] == 3:
                        # Special case: [B, 1, 3, H, W] -> [B, 3, H, W]
                        x = x.squeeze(1)

                # Verify tensor shape before passing to model
                if x.dim() != 4 or x.shape[1] != 3:
                    raise ValueError(f"Input tensor has incorrect shape: {x.shape}. Expected [B, 3, H, W]")

                # Process with ViT-MAE model
                outputs = self.net(x, output_hidden_states=True)
                # Get the [CLS] token from the last hidden state
                cls_token = outputs.last_hidden_state[:, 0]
                # Pass through classifier to get logits
                logits = self.classifier(cls_token)
                # For binary classification, we need to return a single value per sample
                # Take the second column (index 1) which represents the probability of class 1
                # and apply sigmoid to get a probability
                x = torch.sigmoid(logits[:, 1]).unsqueeze(1)  # Shape: [batch_size, 1]
            else:
                # For other models
                x = self.net(x)
            return x
        except Exception as e:
            # Provide more detailed error information
            error_msg = f"Error in forward pass: {str(e)}. Input tensor shape: {x.shape}, dtype: {x.dtype}, min: {x.min().item()}, max: {x.max().item()}"
            raise RuntimeError(error_msg) from e

    def training_step(self,x,target):
        # Note: We don't need to apply label smoothing here anymore
        # as it's handled by the loss function itself if needed

        for i in range(2):
            pred_cls=self(x)
            if i==0:
                pred_first=pred_cls
            loss_cls=self.cel(pred_cls,target)
            loss=loss_cls
            self.optimizer.zero_grad()
            loss.backward()
            if i==0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first
