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

nn.DiceLoss = DiceLoss # surcharger la los de base

def get_loss(lossname):
    return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim
