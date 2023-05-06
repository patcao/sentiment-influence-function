from __future__ import annotations

from pathlib import Path
from typing import List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def get_train_df() -> pd.DataFrame:
    data_path = Path("data") / "train.csv"
    return pd.read_csv(data_path)


def get_test_df() -> pd.DataFrame:
    data_path = Path("data") / "val.csv"
    return pd.read_csv(data_path)


def create_loo_dataset(sst2_dataset: Dataset, loo_guids: Union[int, List[int]]):
    guids, inputs, masks, labels = sst2_dataset.tensors

    loo_mask = torch.zeros_like(guids, dtype=torch.bool)

    try:
        iterator = iter(loo_guids)
    except TypeError:
        iterator = iter([loo_guids])

    for val in iterator:
        loo_mask |= guids == val
    loo_mask = ~loo_mask

    return TensorDataset(
        guids[loo_mask], inputs[loo_mask], masks[loo_mask], labels[loo_mask]
    )


def get_train_example(guid: int):
    train_df = get_train_df()
    return train_df[train_df.guid == guid]


def get_test_example(guid: int):
    test_df = get_test_df()
    return test_df[test_df.guid == guid]


def get_tokens_from_ids(input_ids, bert_name: str = "distilbert-base-uncased"):
    tokenizer = AutoTokenizer.from_pretrained(bert_name, do_lower_case=True)

    trimmed_input_ids = []
    for input_id in input_ids:
        trimmed_input_ids.append(input_id)
        if input_id == 102:
            break

    return tokenizer.convert_ids_to_tokens(trimmed_input_ids)


def create_train_sst2(
    num_samples: int = -1,
    tokenizer_name: str = "distilbert-base-uncased",
    max_seq_len: int = 64,
    use_bert_embeddings=False,
    device=None,
) -> Dataset:
    """guid, inputs, masks, labels"""
    data_path = Path("data") / "train.csv"
    train_df = pd.read_csv(data_path)
    return create_sst2_dataset(
        train_df,
        num_samples,
        tokenizer_name,
        max_seq_len,
        device=device,
        use_bert_embeddings=use_bert_embeddings,
    )


def create_test_sst2(
    num_samples: int = -1,
    tokenizer_name: str = "distilbert-base-uncased",
    max_seq_len: int = 64,
    use_bert_embeddings=False,
    device=None,
) -> Dataset:
    """guid, inputs, masks, labels"""
    data_path = Path("data") / "val.csv"
    test_df = pd.read_csv(data_path)
    return create_sst2_dataset(
        test_df,
        num_samples,
        tokenizer_name,
        max_seq_len,
        device=device,
        use_bert_embeddings=use_bert_embeddings,
    )


def create_sst2_dataset(
    df: pd.DataFrame,
    num_samples: int = -1,
    tokenizer_name: str = None,
    max_seq_len: int = 64,
    use_bert_embeddings=False,
    device=None,
) -> TensorDataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

    if num_samples != -1:
        df = df.iloc[:num_samples]
        df = df.reset_index(drop=True)

    inputs, masks = _preprocessing_for_bert(
        data=df.sentence,
        max_length=max_seq_len,
        bert_tokenizer=tokenizer,
        truncation=True,
    )
    guids = torch.IntTensor(df.guid)
    labels = torch.LongTensor(df.label)
    inputs, masks = inputs, masks

    # Use the BERT embedding for each token instead of the id
    if use_bert_embeddings:
        bert_pretrained = AutoModel.from_pretrained(tokenizer_name)
        word_embeddings = bert_pretrained.get_input_embeddings()
        inputs = word_embeddings(inputs).detach().clone()

    if device is not None:
        guids = guids.to(device)
        inputs = inputs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
    dataset = TensorDataset(guids, inputs, masks, labels)

    # TODO very hacky, subclass Dataset instead
    dataset.use_bert_embeddings = use_bert_embeddings
    return dataset


def _preprocessing_for_bert(data, max_length, bert_tokenizer, **kwargs):
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
        device,
        df: pd.DataFrame,
        tokenizer_name: str,
        max_seq_len: int = 64,
    ) -> SST2Dataset:
        # Keep only a fraction of the data
        # keep_len = math.floor(frac * len(df))
        # df = df.iloc[:keep_len]

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=True)

        inputs, masks = cls._preprocessing_for_bert(
            df.sentence, max_seq_len, tokenizer, truncation=True
        )
        guids = torch.IntTensor(df.guid).to(device)
        labels = torch.LongTensor(df.label).to(device)
        inputs, masks = inputs.to(device), masks.to(device)
        return TensorDataset(inputs, masks, labels)
        # return SST2Dataset(df, guids, inputs, masks, labels)

    def __init__(
        self,
        df: pd.DataFrame,
        guids: torch.Tensor,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ):
        self.df = df
        self.guids = guids
        self.inputs = inputs
        self.masks = masks
        self.labels = labels

    def leave_one_out(self, leave_out_guid: int):
        # Keep everything except the leave out guid data point
        tensor_mask = ~(self.guids == leave_out_guid)

        return SST2Dataset(
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
