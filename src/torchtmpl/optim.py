# coding: utf-8

# External imports
import torch
import torch.nn as nn
import segmentation_models_pytorch.losses as smp_losses 

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = smp_losses.DiceLoss(mode='binary', from_logits=True)

    def forward(self, y_pred, y_true):
        return self.loss_fn(y_pred, y_true)

nn.DiceLoss = DiceLoss # surcharger la loss de base

class DiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=0.5):
        super().__init__()
        self.dice = smp_losses.DiceLoss(mode="binary", from_logits=True)
        self.focal = smp_losses.FocalLoss(mode="binary", gamma=2.0)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, outputs, targets):
        # On calcule les deux pertes
        loss_dice = self.dice(outputs, targets)
        loss_focal = self.focal(outputs, targets)
        
        # On somme les résultats (des tenseurs) pondérés
        return (self.dice_weight * loss_dice) + (self.focal_weight * loss_focal)


def get_loss(lossname):
    if lossname == "TverskyLoss":
        return smp_losses.TverskyLoss(mode="binary", alpha=0.7, beta=0.3, from_logits=True)
    
    elif lossname == "FocalLoss":
        return smp_losses.FocalLoss(mode="binary", gamma=2.0)
        
    elif lossname == "DiceLoss":
        return smp_losses.DiceLoss(mode="binary", from_logits=True)

    # C'est ici qu'on appelle notre nouvelle classe
    elif lossname == "DiceFocalLoss":
        return DiceFocalLoss(dice_weight=0.8, focal_weight=0.5)

    return eval(f"nn.{lossname}()")



def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
