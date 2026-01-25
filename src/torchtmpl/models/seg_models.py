# coding: utf-8
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# --- CLASSES WRAPPERS POUR SMP ---

class Unet(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        
        encoder_name = cfg.get("backbone", "resnet34")
        encoder_weights = "imagenet"
        in_channels = input_size[0] 

        self.model = smp.Unet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=in_channels,          
            classes=num_classes,              
            activation=None 
        )

    def forward(self, x):
        return self.model(x)


class DeepLabV3(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        
        encoder_name = cfg.get("backbone", "resnet34")
        encoder_weights = "imagenet" 
        in_channels = input_size[0]

        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=in_channels,          
            classes=num_classes,              
            activation=None
        )

    def forward(self, x):
        return self.model(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        
        encoder_name = cfg.get("backbone", "resnet34")
        encoder_weights = "imagenet"
        in_channels = input_size[0]

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=in_channels,          
            classes=num_classes,              
            activation=None
        )

    def forward(self, x):
        return self.model(x)

class UnetPlusPlus(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        
        encoder_name = cfg.get("backbone", "resnet34")
        encoder_weights = "imagenet" if cfg.get("pretrained", True) else None
        in_channels = input_size[0]

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            in_channels=in_channels,          
            classes=num_classes,              
            activation=None
        )

    def forward(self, x):
        return self.model(x)
