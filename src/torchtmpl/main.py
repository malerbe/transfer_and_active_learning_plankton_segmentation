# coding: utf-8

# Standard imports
import logging
import sys
import os
import pathlib
import itertools
import yaml
import json
import copy

# External imports
import yaml
import wandb
import torch
import torchinfo.torchinfo as torchinfo

# Local imports
from . import data
from . import models
from . import optim
from . import utils


def train(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Using device {device}")

    if "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        wandb.init(project=wandb_config["project"], entity=wandb_config["entity"])
        wandb_log = wandb.log
        wandb_log(config)
        logging.info(f"Will be recording in wandb run name : {wandb.run.name}")
    else:
        wandb_log = None

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
        data_config, use_cuda
    )

    # Build the model
    logging.info("= Model")
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    # Build the loss
    logging.info("= Loss")
    loss = optim.get_loss(config["loss"])

    # Build the optimizer
    logging.info("= Optimizer")
    optim_config = config["optim"]
    optimizer = optim.get_optimizer(optim_config, model.parameters())

    # Build the callbacks
    logging_config = config["logging"]
    # Let us use as base logname the class name of the modek
    logname = model_config["class"]
    logdir = utils.generate_unique_logpath(logging_config["logdir"], logname)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    logging.info(f"Will be logging into {logdir}")

    # Copy the config file into the logdir
    logdir = pathlib.Path(logdir)
    with open(logdir / "config.yaml", "w") as file:
        yaml.dump(config, file)

    # Make a summary script of the experiment
    images, _ = next(iter(train_loader))
    device = next(model.parameters()).device
    images = images.to(device)
    model_summary = torchinfo.summary(model, input_data=images)
    summary_text = (
        f"Logdir : {logdir}\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + f" Config : {config} \n\n"
        + (f" Wandb run name : {wandb.run.name}\n\n" if wandb_log is not None else "")
        + "## Summary of the model architecture\n"
        + f"{model_summary}\n\n"
        + "## Loss\n\n"
        + f"{loss}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )
    with open(logdir / "summary.txt", "w", encoding='utf-8') as f:
        f.write(summary_text)
    logging.info(summary_text)
    if wandb_log is not None:
        wandb.log({"summary": summary_text})

    logging.info("= Building lr-scheduler")
    patience = 3
    factor = 0.5
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience
    )

    # Define the early stopping callback
    model_checkpoint = utils.ModelCheckpoint(
        model, str(logdir / "best_model.pt"), min_is_best=True
    )

    for e in range(config["nepochs"]):
        # Train 1 epoch
        train_loss = utils.train(model, train_loader, loss, optimizer, device)

        # Test
        test_loss, test_dice, test_iou = utils.test(model, valid_loader, loss, device, num_classes)

        # Step Scheduler based on validation loss
        scheduler.step(test_loss)
        
        # Checkpoint update
        is_best = model_checkpoint.update(test_loss)
        
        # Logging console
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(
                "[%d/%d] Train Loss: %.5f | Test Loss: %.5f | Dice: %.4f | IoU: %.4f | LR: %.1e %s"
                % (
                    e + 1,
                    config["nepochs"],
                    train_loss,
                    test_loss,
                    test_dice,
                    test_iou,
                    current_lr,
                    "[>> BEST <<]" if is_best else "",
                )
            )

        # Update the dashboard
        metrics = {
            "train_loss": train_loss, 
            "test_loss": test_loss,
            "test_dice": test_dice,
            "test_iou": test_iou
        }
        if wandb_log is not None:
            wandb_log(metrics)

def train_transfer_learning_gridsearch(config):
    """
    This function allows training the base model learned using on the external dataset and validate using the goal
    dataset. It is a single training but part of a bigger pipeline made of two parts:
        - Model hyperparameters tuning: find the right hyperparameters which will allow the best performance on
        the validation set (made of samples from the goal dataset) (this function)
        - Training on differents sets of transformations to be able to compare the effect of different transformations
        both on a validation set taken from the external dataset and an another validation set taken from the goal
        dataset (train_transfer_learning_optimize_transformations)
    """

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")
    print(f"Using device {device}")

    ##########################################################
    # GridSearch on the hyperparameters
    ##########################################################

    # Build the dataloaders
    logging.info("= Building the dataloaders")
    data_config = config["data"]

    # Gridsearch de base
    # grid_params = {
    #     'learning_rate': [0.001, 0.0001, 0.0005],
    #     'model_backbone': ['resnet34'],
    #     'model': ['Unet', 'DeepLabV3Plus'],
    #     'freeze': [True, False],
    #     'weight_decay': [1e-4, 1e-2],
    #     'augmentations': ['basic']
    # }

    # Gridsearch augmentations
    grid_params = {
        'learning_rate': [0.001, 0.0001],
        'model_backbone': ['resnet34'],
        'model': ['Unet', 'DeepLabV3Plus'],
        'freeze': [True, False],
        'weight_decay': [1e-2],
        # 'augmentations': ["tl_occlusion_affine_domain_fda"]
        # 'augmentations': ["tl_basic", "tl_occlusion", "tl_occlusion_affine", "tl_occlusion_affine_domain", "tl_occlusion_affine_domain_fda"]
        'augmentations': ["tl_occlusion_affine_domain", "tl_occlusion_affine_domain_fda"]
    
    }


    # grid_params = {
    #     'learning_rate': [0.001],
    #     'model_backbone': ['resnet34'],
    #     'model': ['Unet'],
    #     'freeze': [True],
    #     'weight_decay': [1e-4],
    #     'augmentations': ["tl_occlusion_affine_domain_fda"]
    # }
    

    # grid_params = {
    #     'learning_rate': [0.001],
    #     'model_backbone': ['resnet34'],
    #     'model': ['Unet'],
    #     'freeze': [True],
    #     'weight_decay': [1e-4]
    # }

    keys, values = zip(*grid_params.items())
    combinations = list(itertools.product(*values))
    logging.info(f"GridSearch: {len(combinations)} combinations to test.")

    all_results = [] # will contain the result for each hyperparameters combination

    # Root folder for gridsearch
    base_logdir = config["logging"]["logdir"]
    grid_search_dir = utils.generate_unique_logpath(base_logdir, f"GridSearch")
    if not os.path.isdir(grid_search_dir):
        os.makedirs(grid_search_dir)

    #### Gridsearch loop
    for i, combo in enumerate(combinations):
        current_params = dict(zip(keys, combo))

        run_config = copy.deepcopy(config)

        run_config["optim"]["lr"] = current_params["learning_rate"]
        run_config["model"] = {}
        run_config["model"]["class"] = current_params["model"]
        run_config["model"]["backbone"] = current_params["model_backbone"]
        run_config["optim"]["params"] = {}
        run_config["optim"]["params"]["weight_decay"] = current_params["weight_decay"] 
        run_config["data"]["augmentations"] = current_params["augmentations"]

        logging.info(f"\n=== GridSearch Run {i+1}/{len(combinations)} with params: {current_params} ===")

        # create folder for this specific run
        run_name = "_".join([f"{k}_{v}" for k, v in current_params.items()]) if current_params else "default"
        run_name = run_name.replace("_", "__").replace(".", "_")[:100]
        run_logdir = os.path.join(grid_search_dir, f"run_{i}_{run_name}")
        if not os.path.isdir(run_logdir):
            os.makedirs(run_logdir)

        # save run configuratio
        with open(os.path.join(run_logdir, "config.yaml"), "w") as file:
            yaml.dump(run_config, file)

        # Build dataloader
        run_config["data"]["transfer_learning_phase"] = "gridsearch"
        train_loader, valid_loader, input_size, num_classes = data.get_dataloaders(
            run_config["data"], use_cuda
        )

        logging.info("= Building Model")
        model = models.build_model(run_config["model"], input_size, num_classes)
        
        # freezing 
        if current_params["freeze"]:
            if hasattr(model, 'encoder'):
                for param in model.encoder.parameters():
                    param.requires_grad = False
                logging.info(">> Encoder is FROZEN")
            else:
                logging.warning("Could not freeze encoder: attribute 'encoder' not found.")
        else:
             logging.info(">> Encoder is UN-FROZEN (Fine-tuning)")
             
        model.to(device)

        logging.info("= Building Loss")
        loss_fn = optim.get_loss(run_config["loss"])

        logging.info("= Building Optimizer")
        optimizer = optim.get_optimizer(run_config["optim"], model.parameters())

        logging.info("= Building lr-scheduler")
        patience = 3
        factor = 0.5
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=factor, patience=patience
        )

        # checkpoint helper
        model_checkpoint = utils.ModelCheckpoint(
            model, os.path.join(run_logdir, "best_model.pt"), min_is_best=True
        )

        best_run_metrics = {}

        epoch_history = {
            "epochs": [],
            "train_loss": [],
            "test_loss": [],
            "test_dice": [],
            "test_iou": [],
            "learning_rate": []
        }

        for e in range(run_config["nepochs"]):
            # Train
            train_loss = utils.train(model, train_loader, loss_fn, optimizer, device)

            # Validate
            test_loss, test_dice, test_iou = utils.test(model, valid_loader, loss_fn, device, num_classes)

            # Step Scheduler based on validation loss
            scheduler.step(test_loss)
            
            # Checkpoint update
            is_best = model_checkpoint.update(test_loss)

            # Save metrics for future analysis
            current_lr = optimizer.param_groups[0]['lr']
            epoch_history["epochs"].append(e + 1)
            epoch_history["train_loss"].append(float(train_loss))
            epoch_history["test_loss"].append(float(test_loss))
            epoch_history["test_dice"].append(float(test_dice))
            epoch_history["test_iou"].append(float(test_iou))
            epoch_history["learning_rate"].append(float(current_lr))
            
            # Logging console
            logging.info(
                "[%d/%d] Train Loss: %.5f | Test Loss: %.5f | Dice: %.4f | IoU: %.4f | LR: %.1e %s"
                % (
                    e + 1,
                    run_config["nepochs"],
                    train_loss,
                    test_loss,
                    test_dice,
                    test_iou,
                    current_lr,
                    "[>> BEST <<]" if is_best else "",
                )
            )

            # If best model:
            if is_best:
                best_run_metrics = {
                    "epoch": e + 1,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "test_dice": test_dice,
                    "test_iou": test_iou
                }

        history_path = os.path.join(run_logdir, "epoch_history.json")
        with open(history_path, "w", encoding='utf-8') as f:
            json.dump(epoch_history, f, indent=4)

        # Save all results
        result_entry = {
            "run_id": i,
            "params": current_params,
            "best_metrics": best_run_metrics,
            "logdir": str(run_logdir)
        }
        all_results.append(result_entry)

        # Empty GPU from the model to be able to instanciate a new one
        del model, optimizer, scheduler, train_loader, valid_loader
        torch.cuda.empty_cache()

    ##########################################################
    # Save global results (json)
    ##########################################################
    json_path = os.path.join(grid_search_dir, "grid_search_results.json")

    def default_serializer(obj):
        if isinstance(obj, (torch.Tensor)):
            return obj.item()
        if hasattr(obj, 'tolist'): # Numpy arrays
            return obj.tolist()
        return str(obj)

    try:
        with open(json_path, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=4, default=default_serializer)
        logging.info(f"GridSearch completed. Results saved to {json_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON results: {e}")

    return grid_search_dir

def test(config):
    raise NotImplementedError


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 3:
        logging.error(f"Usage : {sys.argv[0]} config.yaml <train|test>")
        sys.exit(-1)

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(sys.argv[1], "r"))

    command = sys.argv[2]
    eval(f"{command}(config)")
