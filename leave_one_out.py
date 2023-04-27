import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.BertClassifier as BertClassifier
import src.utils as utils
import wandb
from src.datasets import (create_loo_dataset, create_test_sst2,
                          create_train_sst2)


def evaluate_loss_df(model, dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    test_losses = []
    # For each batch in our validation set...
    for batch in dataloader:
        # Load batch to GPU
        b_guids, b_input_ids, b_attn_mask, b_labels = batch

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        if hasattr(logits, "logits"):
            logits = logits.logits

        # Compute loss
        loss = loss_fn(logits, b_labels)
        test_losses.append(
            {"guid": b_guids.item(), "label": b_labels.item(), "loss": loss.item()}
        )

    return pd.DataFrame(test_losses)


def main(args):
    device = utils.get_device()
    loo_guid = 0

    with open(args.config_path, "r") as stream:
        config = yaml.safe_load(stream)
    config.update(
        {"epochs": args.epochs, "num_training_examples": args.num_training_examples}
    )

    # Create datasets
    train_dataset = create_train_sst2(
        device,
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )

    test_dataset = create_test_sst2(
        device,
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    # print(f"Train: {len(train_dataloader)*config['batch_size']}")
    # print(f"Test: {len(test_dataloader)}")

    for loo_guid in range(args.loo_guid_start, args.loo_guid_end):
        config["loo_guid"] = loo_guid
        run = wandb.init(project="LOO-BertClassifier", config=config)

        # Create train dataset
        loo_dataset = create_loo_dataset(train_dataset, loo_guid)
        train_dataloader = DataLoader(
            loo_dataset, batch_size=config["batch_size"], shuffle=True
        )

        # Create classifcation model
        model = BertClassifier.create_bert_classifier(
            config["bert_model_name"],
            classifier_type=config["classifier_type"],
            classifier_hidden_size=config["classifier_hidden_size"],
            classifier_drop_out=config["classifier_drop_out"],
            freeze_bert=True,
            random_state=42,
        )

        # Do training
        optimizer = Adam(model.classifier.parameters(), lr=config["learning_rate"])
        loss_fn = torch.nn.CrossEntropyLoss()

        timings = utils.train(
            config=config,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            train_dataloader=train_dataloader,
            val_dataloader=None,
        )

        test_loss, test_acc = utils.evaluate(model, test_dataloader)
        wandb.summary["test/loss"] = test_loss
        wandb.summary["test/accuracy"] = test_acc

        wandb.finish()

        # Save model parameters
        output_dir = Path(args.output_dir) / f"run_{loo_guid}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.classifier.state_dict(), output_dir / "classifier_params.pt")

        # Compute loss for each test point
        df = evaluate_loss_df(model, test_dataloader)
        df.to_csv(output_dir / "test_loss.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to output LOO results"
    )
    parser.add_argument(
        "--config-path", type=str, help="Path to the hyperparameter config YAML file"
    )
    parser.add_argument(
        "--loo-guid-start",
        type=int,
        help="GUID of the first training example to leave out, inclusive",
    )
    parser.add_argument(
        "--loo-guid-end",
        type=int,
        help="GUID of the last training example to leave out, exclusive",
    )
    parser.add_argument(
        "--num-training-examples", type=int, help="Number of training examples to use"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train model for"
    )

    args = parser.parse_args()
    main(args)
