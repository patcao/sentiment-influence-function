import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.BertClassifier as BertClassifier
import src.utils as utils
import wandb
from src.datasets import create_test_sst2, create_train_sst2


def main_train_loop():
    config = wandb.config
    device = utils.get_device()

    # Create datasets
    train_dataset = create_train_sst2(
        device,
        num_samples=config.num_training_examples,
        tokenizer_name=config.bert_model_name,
        max_seq_len=config.max_sequence_length,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    test_dataset = create_test_sst2(
        device,
        tokenizer_name=config.bert_model_name,
        max_seq_len=config.max_sequence_length,
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False)

    print(f"Train: {len(train_dataloader)*config.batch_size}")
    print(f"Test: {len(test_dataloader)}")

    # Create classifcation model
    model = BertClassifier.create_bert_classifier(
        config.bert_model_name,
        classifier_type=config.classifier_type,
        classifier_hidden_size=config.classifier_hidden_size,
        classifier_drop_out=config.classifier_drop_out,
        freeze_bert=True,
        random_state=42,
    )

    # Do training
    optimizer = Adam(model.classifier.parameters(), lr=config.learning_rate)
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


run = wandb.init()
# run = wandb.init(
#     project="BertClassifier",
#     group="fine-tuning",
#     tags=["Adam", "single-fc"],
#     config={
#         "epochs": 20,
#         "num_training_examples": 16,
#         "bert_model_name": "distilbert-base-uncased",
#         "max_sequence_length": 64,
#         "learning_rate": 5e-3,
#         "lr_warmup_pct": 0.1,
#         "batch_size": 16,
#         "classifier_type": "double-fc",
#         "classifier_hidden_size": 100,
#         "classifier_drop_out": 0,
#     },
# )
main_train_loop()
