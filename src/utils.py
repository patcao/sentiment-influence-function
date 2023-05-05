import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score
# from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

import wandb


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")

    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")

    # Plot ROC AUC
    plt.title("Receiver Operating Characteristic")
    plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.show()


def get_auc_CV(model, X_train_tfidf, y_train):
    """
    Return the average AUC score from cross-validation.
    """
    # Set KFold to shuffle data before the split
    kf = StratifiedKFold(5, shuffle=True, random_state=1)

    # Get AUC scores
    auc = cross_val_score(model, X_train_tfidf, y_train, scoring="roc_auc", cv=kf)

    return auc.mean()


def evaluate_loss_df(model, dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    device = get_device()

    model.to(device)

    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

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
        if hasattr(logits, "logits"):
            logits = logits.logits

        # Compute loss
        loss = loss_fn(logits, b_labels)

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
        set_seed(random_state)  # Set seed for reproducibility

    device = get_device()
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
                b_guids, b_input_ids, b_attn_mask, b_labels = tuple(
                    t.to(device) for t in batch
                )

                model.zero_grad()
                logits = model(b_input_ids, b_attn_mask)
                loss = model.compute_loss(logits, b_labels)
                batch_loss = loss.item()
                total_loss += batch_loss

                # Get the predictions
                preds = torch.argmax(logits, dim=1).flatten()
                accuracy = (preds == b_labels).cpu().numpy().mean() * 100
                total_acc += accuracy

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
            _, val_loss, val_acc = evaluate_loss_df(model, val_dataloader)
            metrics["val/loss"] = val_loss
            metrics["val/accuracy"] = val_acc

        wandb.log(metrics)
