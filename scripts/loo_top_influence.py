import os
import sys

# Put the src directory on the path
# TODO could do in more reliable manner
sys.path.insert(0, os.getcwd())

import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src import BertClassifier, influence, train_utils, utils
from src.datasets import create_loo_dataset, create_test_sst2, create_train_sst2

"""Performs Leave One Out retraining. 
This retrains each model from the best parameters
"""


def pick_top_influence_guids(args):
    idf = pd.read_csv(f"{args.influence_dir}/influence-testguid-{args.test_guid}.csv")
    idf["abs_influence"] = np.abs(idf["influence"])

    # Pick training IDS with top 100
    loo_indxs = (
        idf.sort_values("abs_influence", ascending=False)
        .iloc[: args.num_influence_points]
        .train_guid.to_list()
    )
    return loo_indxs


def pick_very_tenth_influence_guids(args):
    idf = pd.read_csv(f"{args.influence_dir}/influence-testguid-{args.test_guid}.csv")
    # idf["abs_influence"] = np.abs(idf["influence"])

    skip = int(10000 / args.num_influence_points)
    # Pick training IDS with top 100
    loo_indxs = (
        idf.sort_values("influence", ascending=False)
        .iloc[::skip, :]
        .train_guid.to_list()
    )
    return loo_indxs


def do_loo_and_save(
    config_path, output_dir, train_dataset, test_dataset, config, loo_guid
):
    # Create train dataset
    loo_dataset = create_loo_dataset(train_dataset, loo_guid)
    loo_dataloader = DataLoader(
        loo_dataset, batch_size=config["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    loo_model, _ = BertClassifier.load_model(config_path)

    if "optimizer_weight_decay" in config:
        optimizer = Adam(
            loo_model.classifier.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["optimizer_weight_decay"],
        )
    else:
        optimizer = Adam(loo_model.classifier.parameters(), lr=config["learning_rate"])

    run = wandb.init(project="Bert-LOO", config=config)
    train_utils.train(
        config=config,
        model=loo_model,
        optimizer=optimizer,
        train_dataloader=loo_dataloader,
        val_dataloader=test_dataloader,
    )

    ldf, test_loss, test_acc = train_utils.evaluate_loss(loo_model, test_dataloader)

    wandb.summary["test/loss"] = test_loss
    wandb.summary["test/accuracy"] = test_acc
    wandb.finish()

    ldf["loo_guid"] = loo_guid
    torch.save(loo_model.classifier.state_dict(), output_dir / "trained.pt")
    ldf.to_csv(output_dir / "test_loss.csv", index=False)


def main(args):
    base_output_dir = Path(args.output_dir) / f"test-guid-{args.test_guid}"
    os.makedirs(base_output_dir, exist_ok=True)

    device = utils.get_device()
    # Load model and config
    # _, config = BertClassifier.load_model(args.config_path)
    config = utils.load_config(args.config_path)
    
    loo_indxs = pick_top_influence_guids(args)
    l = loo_indxs[2:]
    l.reverse()
    l = [761, 7127] + l

    # loo_indxs = pick_very_tenth_influence_guids(args)
    worker_id = args.worker_id
    work_split = utils.split_list(loo_indxs, args.num_workers)
    work_split = work_split[worker_id - 1]

    str_work_split = [str(val) for val in work_split]
    config.update(
        worker_id=worker_id,
        num_workers=args.num_workers,
        # work_split=",".join(str_work_split),
    )
    retrain_config = config.copy()
    if args.learning_rate is not None:
        retrain_config.update(epochs=args.epochs, learning_rate=args.learning_rate)
    else:
        retrain_config.update(epochs=args.epochs, learning_rate=0.0001)

    utils.save_config(retrain_config, f"{base_output_dir}/worker-{worker_id}.yaml")

    # Create datasets
    train_dataset = create_train_sst2(
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=device,
    )

    test_dataset = create_test_sst2(
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=device,
    )
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)

    for loo_guid in work_split:
        retrain_config["loo_guid"] = loo_guid

        # Create LOO directory
        output_dir = base_output_dir / f"run_{loo_guid}"
        os.makedirs(output_dir, exist_ok=True)

        do_loo_and_save(
            config_path=args.config_path,
            output_dir=output_dir,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            config=retrain_config,
            loo_guid=loo_guid,
        )


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
        "--influence-dir", type=str, help="Directory to get influence results"
    )
    parser.add_argument(
        "--test-guid", type=int, help="Test guid to perform the LOO experiments for"
    )
    parser.add_argument(
        "--num-influence-points",
        type=int,
        help="Includes the top N influence points",
        default=100,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to retrain the LOO models for",
        default=3,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Learning rate to user when retraining",
        default = None,
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

    args = parser.parse_args()
    main(args)
