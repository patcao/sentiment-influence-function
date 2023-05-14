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

import wandb
from src import BertClassifier, train_utils, utils
from src.datasets import create_loo_dataset, create_test_sst2, create_train_sst2

"""Performs Leave One Out retraining. This retrains each model from the start for every LOO"""


def main(args):
    device = utils.get_device()

    config = utils.load_config(args.config_path)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.num_training_examples:
        config["num_training_examples"] = args.num_training_examples

    worker_id = args.worker_id
    all_loo_guids = list(range(config["num_training_examples"]))
    work_split = utils.split_list(all_loo_guids, args.num_workers)
    work_split = work_split[worker_id - 1]

    str_work_split = [str(val) for val in work_split]
    config.update(worker_id=worker_id, work_split=','.join(str_work_split))

    os.makedirs(args.output_dir, exist_ok=True)
    utils.save_config(config, f"{args.output_dir}/worker-{worker_id}.yaml")

    # Create datasets
    train_dataset = create_train_sst2(
        device=device,
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )

    test_dataset = create_test_sst2(
        device=device,
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
    )

    for loo_guid in work_split:
        config["loo_guid"] = loo_guid

        # Create LOO directory
        output_dir = Path(args.output_dir) / f"run_{loo_guid}"
        os.makedirs(output_dir, exist_ok=True)

        # Create train dataset
        loo_dataset = create_loo_dataset(train_dataset, loo_guid)

        model, df, test_loss, test_acc = train_utils.train_bert_model(
            train_dataset=loo_dataset,
            test_dataset=test_dataset,
            config=config,
            wandb_project="LOO-Bert",
        )

        torch.save(model.classifier.state_dict(), output_dir / "trained.pt")
        df.to_csv(output_dir / "test_loss.csv", index=False)


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
        "--worker-id", type=int, help="ID of this worker. Starts at 1", default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="GUID of the last training example to leave out, exclusive",
        default=1,
    )
    parser.add_argument(
        "--num-training-examples", type=int, help="Number of training examples to use"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of epochs to train model for"
    )

    args = parser.parse_args()
    main(args)
