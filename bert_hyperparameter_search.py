import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.BertClassifier as BertClassifier
import src.utils as utils
import wandb
from src import train_utils
from src import datasets as data_utils

def main_train_loop():
    config = wandb.config
    device = utils.get_device()

    # Create datasets
    train_dataset = data_utils.create_train_sst2(
        device=device,
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )

    test_dataset = data_utils.create_test_sst2(
        device=device,
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )

    train_utils.train_bert_model(
        train_dataset, test_dataset, config, validation_dataset=test_dataset
    )


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
