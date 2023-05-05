import os

import numpy as np
import pandas as pd

from src import influence as inf_utils
from src import train_utils, utils
from src.datasets import (create_loo_dataset, create_test_sst2,
                          create_train_sst2)


def centered_percentile_idxs(infl, remove_length):
    half = int(len(infl) / 2)
    start_index = max(0, half - int(remove_length / 2))
    end_index = start_index + remove_length
    return np.argsort(infl)[start_index:end_index]


def compute_loo_sweep(
    full_model,
    config,
    train_dataset,
    test_dataset,
    test_guid: int,
    wandb_project="LOO-pct",
) -> pd.DataFrame:
    loo_dfs = []

    param_infl = list(full_model.classifier.parameters())
    infl = inf_utils.compute_influence(
        full_model=full_model,
        test_guid=test_guid,
        param_influence=param_infl,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        lissa_r=1,
        lissa_depth=0.3,
        damping=5e-3,
        scale=100,
    )
    # np.arange(0.05, 0.8, 0.05)
    pcts = [0.05, 0.1, 0.15, 0.20, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    for remove_pct in pcts:
        remove_length = int(remove_pct * len(train_dataset))

        # Remove random indices
        remove_idxs = np.random.randint(
            low=0, high=len(train_dataset), size=remove_length
        )
        loo_dataset = create_loo_dataset(train_dataset, remove_idxs)
        _, rdf, rad_test_loss, rand_test_acc = train_utils.train_bert_model(
            loo_dataset, test_dataset, config, wandb_project=wandb_project
        )
        rdf["type"] = "rand"

        # Remove top influence score
        remove_idxs = np.argsort(-infl)[:remove_length]
        loo_dataset = create_loo_dataset(train_dataset, remove_idxs)
        _, tdf, rad_test_loss, rand_test_acc = train_utils.train_bert_model(
            loo_dataset, test_dataset, config, wandb_project=wandb_project
        )
        tdf["type"] = "top"

        # Remove bottom influence score
        remove_idxs = np.argsort(infl)[:remove_length]
        loo_dataset = create_loo_dataset(train_dataset, remove_idxs)
        _, bdf, rad_test_loss, rand_test_acc = train_utils.train_bert_model(
            loo_dataset, test_dataset, config, wandb_project=wandb_project
        )
        bdf["type"] = "bot"

        # Remove near 0 influence score
        remove_idxs = centered_percentile_idxs(infl, remove_length)
        loo_dataset = create_loo_dataset(train_dataset, remove_idxs)
        _, zdf, rad_test_loss, rand_test_acc = train_utils.train_bert_model(
            loo_dataset, test_dataset, config, wandb_project=wandb_project
        )
        zdf["type"] = "zero"

        df = pd.concat([rdf, tdf, bdf, zdf], axis=0)
        df["remove_pct"] = remove_pct

        loo_dfs.append(df)
    return pd.concat(loo_dfs)


def main():
    os.environ["WANDB_SILENT"] = "true"
    NUM_TEST_POINTS = 2

    device = utils.get_device()
    output_dir = "loo_pct_10k"

    config = utils.load_config(
        "model_params/bert_classifier.yaml", epochs=5, num_training_examples=1000
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

    full_model, fdf, full_test_loss, full_test_acc = train_utils.train_bert_model(
        train_dataset, test_dataset, config
    )

    # random_test_guids = np.random.randint(
    #     low=0, high=len(test_dataset), size=NUM_TEST_POINTS
    # )
    test_guids = [1, 3, 4, 12, 11, 13]
    # test_guids = [1, 3, 4]
    # test_guids = [12, 11, 13]
    for test_guid in test_guids:
        print(f"----Sweeping test-guid: {test_guid}----")
        df = compute_loo_sweep(
            full_model, config, train_dataset, test_dataset, test_guid
        )
        df.to_csv(f"{output_dir}/{test_guid}-sweep.csv", index=False)


main()
