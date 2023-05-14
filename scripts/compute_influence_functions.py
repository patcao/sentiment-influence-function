import os
import sys

# Put the src directory on the path
# TODO could do in more reliable manner
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import argparse
from src import BertClassifier
from src import influence as inf_utils
from src import train_utils, utils
from src.datasets import create_loo_dataset, create_test_sst2, create_train_sst2
import random
import pickle

LISSA_DEPTH = 0.15
DAMPING_TERM = 5e-3


def compute_and_save_influence(test_guid: int, config_path: str, output_dir: str):
    cuda_device = utils.get_device()
    model, config = BertClassifier.load_model(config_path)

    # Create datasets
    train_dataset = create_train_sst2(
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=cuda_device,
    )

    test_dataset = create_test_sst2(
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=cuda_device,
    )

    param_infl = list(model.classifier.parameters())

    infl = inf_utils.compute_influence(
        full_model=model,
        test_guid=test_guid,
        param_influence=param_infl,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        lissa_r=1,
        lissa_depth=LISSA_DEPTH,
        damping=DAMPING_TERM,
        scale=100,
        wandb_logging=True,
    )
    df = pd.DataFrame(data=infl, index=range(len(infl)), columns=["influence"])
    df = df.rename_axis("train_guid").reset_index()
    df["test_guid"] = test_guid
    df.to_csv(f"{output_dir}/influence-testguid-{test_guid}.csv", index=False)


def main(args):
    handler = utils.GracefulInterruptHandler()

    worker_id = args.worker_id

    exclude_guids = [218, 303, 862]  # , 862, 292, 112, 315, 334, 651, 443, 303]
    #all_test_guids = list(set(range(872)) - set(exclude_guids))
    # all_test_guids = list(range(872))
    # # all_test_guids = [586, 58, 93, 346, 420, 699]
    # random.Random(42).shuffle(all_test_guids)

    with open('test_guid_order_if.pkl', 'rb') as f:
        test_guid_order = pickle.load(f)

    work_split = utils.split_list(test_guid_order, args.num_workers)

    for test_guid in work_split[worker_id - 1]:
        if os.path.exists(f"{args.output_dir}/influence-testguid-{test_guid}.csv"):
            continue
        if handler():
            break
        compute_and_save_influence(test_guid, args.config_path, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to output LOO results",
        default="influence_results",
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

    args = parser.parse_args()
    main(args)
