import os

import numpy as np
import pandas as pd
import argparse
from src import BertClassifier
from src import influence as inf_utils
from src import train_utils, utils
from src.datasets import create_loo_dataset, create_test_sst2, create_train_sst2

def compute_and_save_influence(test_guid: int):
    cuda_device = utils.get_device()
    model, config = BertClassifier.load_model("model_params/bert-classifier-epoch5-1000.pt")

    # Create datasets
    train_dataset = create_train_sst2(
        num_samples=config["num_training_examples"],
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=cuda_device
    )

    test_dataset = create_test_sst2(
        tokenizer_name=config["bert_model_name"],
        max_seq_len=config["max_sequence_length"],
        device=cuda_device
    )


    param_infl = list(model.classifier.parameters())

    infl = inf_utils.compute_influence(
        full_model=model,
        test_guid=test_guid,
        param_influence=param_infl,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        lissa_r=2,
        lissa_depth=1,
        damping=5e-3,
        scale=100,
        wandb_logging=True
    )
    df = pd.DataFrame(data=infl, index=range(len(infl)), columns=["influence"])
    df = df.rename_axis("train_guid").reset_index()
    df["test_guid"] = test_guid
    df.to_csv(f"{args.output_dir}/influence-testguid-{test_guid}", index=False)



def main(args):
    #     p.map(compute_and_save_influence, range(872))
    worker_id = args.worker_id
    all_test_guids = list(range(872))[:10]
    work_split = utils.split_list(all_test_guids, 3)

    for test_guid in work_split[worker_id-1]:
        compute_and_save_influence(test_guid)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to output LOO results",
        default="influence_results"
    )
    parser.add_argument(
        "--worker-id",
        type=int,
        help="ID of this worker. Starts at 1",
        default=1
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="GUID of the last training example to leave out, exclusive",
        default=1
    )

    args = parser.parse_args()
    main(args)

       
    