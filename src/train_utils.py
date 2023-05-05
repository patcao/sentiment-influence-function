import math

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src import BertClassifier, utils


def evaluate_loss(model, dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    device = utils.get_device()

    model.to(device)

    model.eval()

    test_losses = []
    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    for batch in dataloader:
        # Load batch to GPU
        b_guids, b_input_ids, b_attn_mask, b_labels = (t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = model.compute_loss(logits, b_labels)

        # Get the predictions
        pred = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (pred == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
        val_loss.append(loss.cpu().item())

        test_losses.append(
            {
                "test_guid": b_guids.item(),
                "logits": logits.cpu().numpy().squeeze(0),
                "pred": pred.item(),
                "label": b_labels.item(),
                "loss": loss.item(),
            }
        )

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return pd.DataFrame(test_losses), val_loss, val_accuracy


def train_bert_model(
    train_dataset,
    test_dataset,
    config,
    validation_dataset=None,
    wandb_project="Bert-scratch",
    wandb_tags=None,
):
    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    # Create Bert model
    model = BertClassifier.create_bert_classifier(
        config["bert_model_name"],
        classifier_type=config["classifier_type"],
        classifier_hidden_size=config["classifier_hidden_size"],
        classifier_drop_out=config["classifier_drop_out"],
        classifier_init_state_path=config["classifier_init_state_path"],
        freeze_bert=True,
    )

    run = wandb.init(project=wandb_project, tags=wandb_tags, config=config)
    optimizer = Adam(model.classifier.parameters(), lr=config["learning_rate"])

    train(
        config=config,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=validation_dataset,
    )

    fdf, test_loss, test_acc = evaluate_loss(model, test_dataloader)

    wandb.summary["test/loss"] = test_loss
    wandb.summary["test/accuracy"] = test_acc
    wandb.finish()

    return model, fdf, test_loss, test_acc


class BertLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, total_steps, end_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.end_lr = end_lr
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))
        else:
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return max(
                self.end_lr,
                0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - self.end_lr)
                + self.end_lr,
            )


def train(
    config,
    model,
    optimizer,
    train_dataloader,
    val_dataloader=None,
    random_state=None,
):
    if random_state is not None:
        utils.set_seed(random_state)  # Set seed for reproducibility

    device = utils.get_device()
    total_steps = len(train_dataloader) * config["epochs"]

    scheduler = BertLRScheduler(
        optimizer,
        warmup_steps=config["lr_warmup_pct"] * total_steps,
        total_steps=total_steps,
        end_lr=1e-6,
    )

    for epoch_i in range(1, config["epochs"] + 1):
        model.train()

        total_acc, total_loss, batch_loss = 0, 0, 0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch):
                # TODO make this model agnostic
                b_guids, b_input_ids, b_attn_mask, b_labels = tuple(
                    t.to(device) for t in batch
                )

                model.zero_grad()
                logits = model(b_input_ids, b_attn_mask)
                loss = model.compute_loss(logits, b_labels)
                batch_loss = loss.item()
                total_loss += batch_loss
                total_acc += model.compute_accuracy(logits, b_labels)

                # Perform a backward pass to calculate gradients
                loss.backward()

                # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and the learning rate
                optimizer.step()
                scheduler.step()
                wandb.log({"train/batch_loss": batch_loss / len(batch)})

        # Calculate the average loss over the entire training data
        num_batches = len(train_dataloader)
        avg_train_loss = total_loss / num_batches
        avg_train_acc = total_acc / num_batches

        metrics = {
            "train/loss": avg_train_loss,
            "train/accuracy": avg_train_acc,
            "epoch": epoch_i,
        }

        if val_dataloader is not None:
            _, val_loss, val_acc = evaluate_loss(model, val_dataloader)
            metrics["val/loss"] = val_loss
            metrics["val/accuracy"] = val_acc

        wandb.log(metrics)
