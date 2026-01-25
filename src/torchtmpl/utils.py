# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm


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

def compute_segmentation_metrics(outputs, targets, num_classes):
    """
    Compute Dice and IoU for a batch.
    outputs: (B, C, H, W) logits
    targets: (B, H, W) integer indices
    """
    # Convert logits to class predictions
    preds = torch.argmax(outputs, dim=1)  # (B, H, W)
    
    # Init accumulators
    dices = []
    ious = []

    # Iterate over classes (skip background 0 if needed, here we keep all)
    # Note: If you want to ignore background, start range at 1
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        
        # Dice: 2*Inter / (Area_pred + Area_target)
        # Add epsilon to avoid division by zero
        pred_sum = pred_inds.float().sum()
        target_sum = target_inds.float().sum()
        dice = (2. * intersection + 1e-8) / (pred_sum + target_sum + 1e-8)
        dices.append(dice.item())

        # IoU: Inter / Union
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou.item())

    return np.mean(dices), np.mean(ious)

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
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward propagation
            outputs = model(inputs)
            loss = f_loss(outputs, targets)

            # Update Loss
            batch_size = inputs.shape[0]
            total_loss += batch_size * loss.item()
            num_samples += batch_size

            dice, iou = compute_segmentation_metrics(outputs, targets, num_classes)
            total_dice += dice
            total_iou += iou
            num_batches += 1
    
    avg_loss = total_loss / num_samples
    avg_dice = total_dice / num_batches
    avg_iou = total_iou / num_batches

    return avg_loss, avg_dice, avg_iou
