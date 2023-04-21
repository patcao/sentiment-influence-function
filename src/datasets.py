from __future__ import annotations
import math
from enum import Enum
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


class SST2Dataset(Dataset):
    # home = Path(
    #     "/content/gdrive/MyDrive/Columbia/RobustStatistics/sentiment-influence-function/"
    # )
    # Directory where raw csv files are kept
    DATA_DIR = Path("data")
    # Directory where cached tensors are kept
    PT_DIR = Path("data_pt")

    @classmethod
    def create_dataset(
        cls,
        name: str,
        device,
        df: pd.DataFrame,
        tokenizer=None,
        max_seq_len: int = 64,
        frac: float = 1,
    ) -> SST2Dataset:
        # Keep only a fraction of the data
        keep_len = math.floor(frac * len(df))
        df = df.iloc[:keep_len]

        if tokenizer is None:
            # tokenizer = BertTokenizer.from_pretrained(
            #     "bert-base-uncased", do_lower_case=True
            # )
            tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased", do_lower_case=True
            )

        inputs, masks = cls._preprocessing_for_bert(
            df.sentence, max_seq_len, tokenizer, truncation=True
        )
        guids = torch.Tensor(df.guid).to(device)
        labels = torch.LongTensor(df.label).to(device)
        inputs, masks = inputs.to(device), masks.to(device)

        return SST2Dataset(max_seq_len, tokenizer, df, guids, inputs, masks, labels)

    @classmethod
    def save_dataset(df, guids, inputs, masks, labels):
        df.to_parquet(cls.PT_DIR / f"{name}-{frac}-df.pq")
        torch.save(guids, cls.PT_DIR / f"{name}-{frac}-guids.pt")
        torch.save(inputs, cls.PT_DIR / f"{name}-{frac}-inputs.pt")
        torch.save(masks, cls.PT_DIR / f"{name}-{frac}-masks.pt")
        torch.save(labels, cls.PT_DIR / f"{name}-{frac}-labels.pt")

    @classmethod
    def load_dataset(
        cls,
        name: str,
        tokenizer=None,
        max_seq_len: int = 64,
    ) -> SST2Dataset:
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased", do_lower_case=True
            )
        df = pd.read_parquet(cls.PT_DIR / f"{name}-{frac}-df.pq")
        guids = torch.load(cls.PT_DIR / f"{name}-{frac}-guids.pt")
        inputs = torch.load(cls.PT_DIR / f"{name}-{frac}-inputs.pt")
        masks = torch.load(cls.PT_DIR / f"{name}-{frac}-masks.pt")
        labels = torch.load(cls.PT_DIR / f"{name}-{frac}-labels.pt")
        return SST2Dataset(max_seq_len, tokenizer, df, guids, inputs, masks, labels)

    @classmethod
    def _preprocessing_for_bert(cls, data, max_length, bert_tokenizer, **kwargs):
        """Perform required preprocessing steps for pretrained BERT.
        @param    data (np.array): Array of texts to be processed.
        @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
        @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                      tokens should be attended to by the model.
        """
        # Create empty lists to store outputs
        input_ids = []
        attention_masks = []

        # For every sentence...
        for sent in tqdm(data):
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = bert_tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_length,  # Max length to truncate/pad
                padding="max_length",
                # pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True,  # Return attention mask
                **kwargs,
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks

    def __init__(
        self,
        max_seq_len: int,
        tokenizer,
        df: pd.DataFrame,
        guids: torch.Tensor,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        self.df = df
        self.guids = guids
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def leave_one_out(self, leave_out_guid: int):
        # Keep everything except the leave out guid data point
        tensor_mask = ~(self.guids == leave_out_guid)

        return SST2Dataset(
            self.max_seq_len,
            self.tokenizer,
            self.df,
            self.guids[tensor_mask],
            self.inputs[tensor_mask],
            self.masks[tensor_mask],
            self.labels[tensor_mask],
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.inputs[idx], self.masks[idx], self.labels[idx])
