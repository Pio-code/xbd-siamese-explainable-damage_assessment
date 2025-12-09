"""
Model architectures for damage classification.
Supports both torchvision (ImageNet-1K) and timm (ImageNet-22K).

Contains:
- SiameseNetwork: Siamese architecture with shared backbone
- load_model_from_checkpoint: Loading saved models

Supported backbones:
CNN: EfficientNet (B0, B3), ResNet50, ConvNeXt (Tiny, Small)

Transformers: Swin Transformer (Tiny, Small)

LIBRARIES:
  - torchvision: ImageNet-1K (1.2M img, 1K classes) - config.use_timm = False
  - timm: ImageNet-22K (14M img, 21.8K classes) - config.use_timm = True
"""

import torch
import torch.nn as nn
import torchvision.models as models

import timm
  

class SiameseNetwork(nn.Module):
    """
    Siamese Network with shared backbone for processing pre/post disaster image pairs.
    
    """
    def __init__(self, config):
        """
        Initializes Siamese model with backbone and classification head.
        
        Args:
            config: Configuration object containing all parameters (TrainingConfig)
        """
        super(SiameseNetwork, self).__init__()
        
        backbone_name = config.backbone_name
        pretrained = config.pretrained
        num_classes = config.num_classes
        hidden_size = config.hidden_size
        dropout_rate = config.dropout_rate
        use_timm = config.use_timm
        
        if use_timm: 
            print(f"Loading {backbone_name} with TIMM (ImageNet-22K)")
            
            if backbone_name == 'convnext_tiny':
                self.backbone = timm.create_model(
                    'convnext_tiny.fb_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0  # no classification layer 
                )
                num_features = self.backbone.num_features  # 768
                print(f" ConvNeXt-Tiny | Features: {num_features} | Pre-training: ImageNet-22K")
            
            elif backbone_name == 'convnext_small':
                self.backbone = timm.create_model(
                    'convnext_small.fb_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0
                )
                num_features = self.backbone.num_features  # 768
                print(f"ConvNeXt-Small | Features: {num_features} | Pre-training: ImageNet-22K")
            
            # TRANSFORMER BACKBONES
            elif backbone_name == 'swin_tiny':
                self.backbone = timm.create_model(
                    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0,
                    img_size=128  # Interpolate position embeddings for 128×128
                )
                num_features = self.backbone.num_features  # 768
                print(f"  Swin-Tiny | Features: {num_features} | Pre-training: ImageNet-22K")
                print(f"  Position embeddings interpolated: 224×224 → 128×128")
            
            elif backbone_name == 'swin_small':
                self.backbone = timm.create_model(
                    'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
                    pretrained=pretrained,
                    num_classes=0,
                    img_size=128
                )
                num_features = self.backbone.num_features  # 768
                print(f"  Swin-Small | Features: {num_features} | Pre-training: ImageNet-22K")
            
            else:
                # List of supported backbones with timm
                timm_models = ['convnext_tiny', 'convnext_small', 'swin_tiny', 'swin_small']
                raise ValueError(
                    f"Backbone '{backbone_name}' with timm not supported. "
                    f"Available timm backbones: {timm_models}\n"
                    f"Or set use_timm=False to use torchvision."
                )
        

        else:
            print(f"Loading {backbone_name} with TORCHVISION (ImageNet-1K)")
            
            if backbone_name == 'efficientnet_b0':
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                self.backbone = models.efficientnet_b0(weights=weights)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
                print(f"EfficientNet-B0 | Features: {num_features}")
                
            elif backbone_name == 'efficientnet_b3':
                weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
                self.backbone = models.efficientnet_b3(weights=weights)
                num_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
                print(f"EfficientNet-B3 | Features: {num_features}")
                
            elif backbone_name == 'resnet50':
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet50(weights=weights)
                num_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()
                print(f" ResNet50 | Features: {num_features}")

            elif backbone_name == 'convnext_tiny':
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
                self.backbone = models.convnext_tiny(weights=weights)
                num_features = self.backbone.classifier[2].in_features
                self.backbone.classifier = nn.Sequential(
                    self.backbone.classifier[0],  # LayerNorm2d
                    self.backbone.classifier[1]   # Flatten
                )
                print(f" ConvNeXt-Tiny | Features: {num_features}")
            
            elif backbone_name == 'convnext_small':
                weights = models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None
                self.backbone = models.convnext_small(weights=weights)
                num_features = self.backbone.classifier[2].in_features
                self.backbone.classifier = nn.Sequential(
                    self.backbone.classifier[0],  
                    self.backbone.classifier[1]   
                )
                print(f"ConvNeXt-Small | Features: {num_features}")
            
            # TRANSFORMER ARCHITECTURES
            elif backbone_name == 'swin_tiny':
                weights = models.Swin_T_Weights.DEFAULT if pretrained else None
                self.backbone = models.swin_t(weights=weights)
                num_features = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
                print(f"Swin-Tiny | Features: {num_features}")
            
            elif backbone_name == 'swin_small':
                weights = models.Swin_S_Weights.DEFAULT if pretrained else None
                self.backbone = models.swin_s(weights=weights)
                num_features = self.backbone.head.in_features
                self.backbone.head = nn.Identity()
                print(f" Swin-Small | Features: {num_features}")
            
            else:
                supported_models = [
                    'efficientnet_b0', 'efficientnet_b3', 'resnet50', 
                    'convnext_tiny', 'convnext_small',
                    'swin_tiny', 'swin_small', 'swin_base'
                ]
                raise ValueError(
                    f"Backbone '{backbone_name}' with torchvision not supported. "
                    f"Available torchvision backbones: {supported_models}\n"
                    f"Or set use_timm=True to use timm (ImageNet-22K)."
                )

        self.classifier_head = nn.Sequential(
            nn.Linear(num_features * 2, hidden_size), 
            nn.ReLU(),                      
            nn.Dropout(dropout_rate),                
            nn.Linear(hidden_size, num_classes)
        )


    def forward(self, img_pre, img_post):
        """
        Performs forward pass.
        
        Args:
            img_pre (torch.Tensor): Batch of pre-disaster patches
            img_post (torch.Tensor): Batch of post-disaster patches
        
        Returns:
            torch.Tensor: Output logits for classes
        """
        embedding_pre = self.backbone(img_pre)
        embedding_post = self.backbone(img_post)
        combined_embedding = torch.cat((embedding_pre, embedding_post), dim=1)
        output = self.classifier_head(combined_embedding)
        return output


def load_model_from_checkpoint(checkpoint_path, config):
    """
    Loads a model from checkpoint for inference.
    Unified function used to load models saved during training.
    
    Args:
        checkpoint_path (str): Path to model .pth file
        config: Model configuration (with backbone_name, hidden_size, device, etc.)
    
    Returns:
        SiameseNetwork: Loaded model in eval mode
    """

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    model = SiameseNetwork(config)
    
    # Load weights (PyTorch native function)
    model.load_state_dict(checkpoint)
    

    model.to(config.device)
    model.eval()
    
    return model
