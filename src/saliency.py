from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import src.utils as utils


def _register_embedding_list_hook(model, embeddings_list, model_type):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())

    embedding_layer = model.bert.embeddings.word_embeddings

    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle


def _register_embedding_gradient_hooks(model, embeddings_gradients, model_type):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0])

    embedding_layer = model.bert.embeddings.word_embeddings
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook


def saliency_map(model, input_ids, input_mask, labels, model_type="BERT"):
    embeddings_list = []
    handle = _register_embedding_list_hook(model, embeddings_list, model_type)
    embeddings_gradients = []
    hook = _register_embedding_gradient_hooks(model, embeddings_gradients, model_type)

    model.zero_grad()
    model.bert.embeddings.word_embeddings.weight.requires_grad = True

    output = model(input_ids, input_mask)
    _loss = model.compute_loss(output, labels)

    _loss.backward()
    handle.remove()
    hook.remove()
    model.bert.embeddings.word_embeddings.requires_grad = False

    saliency_grad = embeddings_gradients[0].detach().cpu().numpy()
    saliency_grad = np.sum(saliency_grad[0] * embeddings_list[0], axis=1)
    norm = np.linalg.norm(saliency_grad, ord=1)
    #     saliency_grad = [math.fabs(e) / norm for e in saliency_grad]
    saliency_grad = [
        (-e) / norm for e in saliency_grad
    ]  # negative gradient for loss means positive influence on decision
    return saliency_grad


def compute_saliency_map(
    model: nn.Module,
    dataset: Dataset,
    bert_name: str = "distilbert-base-uncased",
):
    """
    Computes the saliency map
    """
    tokenizer = AutoTokenizer.from_pretrained(bert_name, do_lower_case=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    ret_map = {}

    for guids, input_ids, attention_mask, labels in dataloader:
        saliency_scores = saliency_map(model, input_ids, attention_mask, labels)
        test_tok_sal_list = []
        for tok, sal in zip(
            tokenizer.convert_ids_to_tokens(input_ids.view(-1).cpu().numpy()),
            saliency_scores,
        ):
            if tok == "[PAD]":
                break
            test_tok_sal_list.append((tok, sal))

        ret_map[guids.item()] = test_tok_sal_list

    return ret_map
