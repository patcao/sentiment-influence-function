from enum import Enum
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


class SST2Dataset(Dataset):
    def __init__(self, dataset_type: DatasetType, tokenizer, max_seq_len: int = 64):
        home = Path(
            "/content/gdrive/MyDrive/Columbia/RobustStatistics/sentiment-influence-function/"
        )
        data_path = home / "data"

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

        if dataset_type == DatasetType.TRAIN:
            df = pd.read_csv(data_path / "train.csv")
        elif dataset_type == DatasetType.TEST:
            df = pd.read_csv(data_path / "train.csv")
        elif dataset_type == DatasetType.VAL:
            df = pd.read_csv(data_path / "val.csv")

        x_features = df.sentence
        labels = df.label
        inputs, masks = self._preprocessing_for_bert(
            x_features, max_seq_len, truncation=True
        )

        self.inputs = torch.Tensor(inputs)
        self.masks = torch.Tensor(masks)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (self.inputs[idx], self.masks[idx], self.labels[idx])

    def _preprocessing_for_bert(self, data, max_length, **kwargs):
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
        for sent in data:
            # `encode_plus` will:
            #    (1) Tokenize the sentence
            #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
            #    (3) Truncate/Pad sentence to max length
            #    (4) Map tokens to their IDs
            #    (5) Create attention mask
            #    (6) Return a dictionary of outputs
            encoded_sent = self.tokenizer.encode_plus(
                text=sent,  # Preprocess sentence
                add_special_tokens=True,  # Add `[CLS]` and `[SEP]`
                max_length=max_length,  # Max length to truncate/pad
                pad_to_max_length=True,  # Pad sentence to max length
                # return_tensors='pt',           # Return PyTorch tensor
                return_attention_mask=True,  # Return attention mask
                **kwargs
            )

            # Add the outputs to the lists
            input_ids.append(encoded_sent.get("input_ids"))
            attention_masks.append(encoded_sent.get("attention_mask"))

        # Convert lists to tensors
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)

        return input_ids, attention_masks
