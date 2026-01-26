# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm
import torch.nn.functional as F
import numpy as np


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath = savepath
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath)
            self.best_score = score
            return True
        return False

def compute_segmentation_metrics(outputs, targets, num_classes=None):
    """
    Calcule le Dice et l'IoU moyen pour un batch binaire.
    outputs : [Batch, 1, H, W] (doit être binaire 0 ou 1)
    targets : [Batch, 1, H, W] (doit être binaire 0 ou 1)
    """
    # On s'assure qu'on travaille sur des tenseurs plats par image
    # [B, 1, H, W] -> [B, H*W]
    outputs = outputs.view(outputs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    # Intersection : pixels à 1 dans les deux
    intersection = (outputs * targets).sum(dim=1)  # Somme par image
    
    # Unions pour Dice et IoU
    # Somme des pixels à 1 dans l'output + dans la target
    total_pixels = outputs.sum(dim=1) + targets.sum(dim=1)
    
    union = total_pixels - intersection # Pour IoU (A + B - AinterB)

    # --- DICE ---
    # Formule : 2*Inter / (Sum_A + Sum_B)
    # On ajoute 1e-8 pour éviter la division par 0
    dice = (2.0 * intersection) / (total_pixels + 1e-8)
    
    # --- IoU ---
    # Formule : Inter / Union
    iou = (intersection) / (union + 1e-8)

    # On retourne la MOYENNE sur le batch
    return dice.mean().item(), iou.mean().item()


def train(model, loader, f_loss, optimizer, device, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    pbar = tqdm.tqdm(loader, desc="Train", leave=True)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        if len(targets.shape) == 3:
            targets = targets.unsqueeze(1)

        targets = (targets > 0).float()

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        batch_size = inputs.shape[0]
        total_loss += batch_size * loss.item()
        num_samples += batch_size
        pbar.set_description(f"Train loss : {total_loss/num_samples:.4f}")

    return total_loss / num_samples


def test(model, loader, f_loss, device, num_classes):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0
    num_samples = 0
    num_batches = 0

    with torch.no_grad(): 
        pbar = tqdm.tqdm(loader, desc="Test", leave=True)
        for (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            if len(targets.shape) == 3:
                targets = targets.unsqueeze(1)

            targets = (targets > 0).float()

            # Compute the forward propagation
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            probs = torch.sigmoid(outputs)
            preds_mask = (probs > 0.5).float()

            # Update Loss
            batch_size = inputs.shape[0]
            total_loss += batch_size * loss.item()
            num_samples += batch_size

            dice, iou = compute_segmentation_metrics(preds_mask, targets, num_classes)
            total_dice += dice
            total_iou += iou
            num_batches += 1
            
            pbar.set_description(f"Test loss : {total_loss/num_samples:.4f}")
    
    avg_loss = total_loss / num_samples
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    return avg_loss, avg_dice, avg_iou
