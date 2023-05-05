from typing import List

import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import src.utils as utils


def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def unflatten_to_param_dim(x, param_shape_tensor):
    tar_p = []
    ptr = 0
    for p in param_shape_tensor:
        len_p = torch.numel(p)
        tmp = x[ptr : ptr + len_p].view(p.shape)
        tar_p.append(tmp)
        ptr += len_p
    return tar_p


def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv


def get_inverse_hvp_lissa(
    v,
    model,
    device,
    param_influence,
    train_loader,
    damping,
    num_samples,
    recursion_depth,
    scale=1e4,
):
    ihvp = None
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                guids, input_ids, input_mask, label_ids = next(lissa_data_iterator)
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                guids, input_ids, input_mask, label_ids = next(lissa_data_iterator)
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            # segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            model.zero_grad()
            # train_loss = model(input_ids, input_mask, label_ids)
            model_output = model(input_ids, input_mask)
            train_loss = loss_fn(model_output, label_ids)

            hvp = hv(train_loss, param_influence, cur_estimate)
            cur_estimate = [
                _a + (1 - damping) * _b - _c / scale
                for _a, _b, _c in zip(v, cur_estimate, hvp)
            ]
            if (j % 200 == 0) or (j == recursion_depth - 1):
                print(
                    "Recursion at depth %s: norm is %f"
                    % (j, np.linalg.norm(gather_flat_grad(cur_estimate).cpu().numpy()))
                )
        if ihvp == None:
            ihvp = [_a / scale for _a in cur_estimate]
        else:
            ihvp = [_a + _b / scale for _a, _b in zip(ihvp, cur_estimate)]
    return_ihvp = gather_flat_grad(ihvp)
    return_ihvp /= num_samples
    return return_ihvp


def compute_influence(
    full_model: nn.Module,
    test_guid: int,
    param_influence: List,
    train_dataset: Dataset,
    test_dataset: Dataset,
    training_indices=None,
    lissa_r: int = 1,
    lissa_depth: float = 0.25,
    damping=3e-3,
    scale=1e4,
):
    device = utils.get_device()
    influences = np.zeros(len(train_dataset))
    # param_influence = list(full_model.classifier.parameters())

    train_dataloader_lissa = DataLoader(
        train_dataset, batch_size=16, shuffle=True, drop_last=True
    )
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for guid, input_ids, input_mask, label_ids in test_dataloader:
        if guid != test_guid:
            continue

        full_model.eval()

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        # L_TEST gradient
        full_model.zero_grad()
        output = full_model(input_ids, input_mask)
        test_loss = full_model.compute_loss(output, label_ids)
        test_grads = autograd.grad(test_loss, param_influence)

        # IVHP
        full_model.train()

        t = int(len(train_dataloader) * lissa_depth)
        print(f"LiSSA reps: {lissa_r} and num_iterations: {t}")

        inverse_hvp = get_inverse_hvp_lissa(
            test_grads,
            full_model,
            device,
            param_influence,
            train_dataloader_lissa,
            damping=damping,
            scale=scale,
            num_samples=lissa_r,
            recursion_depth=t,
        )

        for train_guid, train_input_id, train_input_mask, train_label in tqdm(
            train_dataloader
        ):
            if training_indices is not None and train_guid not in training_indices:
                continue

            full_model.train()
            full_model.zero_grad()
            train_output = full_model(train_input_id, train_input_mask)
            train_loss = full_model.compute_loss(train_output, train_label)

            train_grads = autograd.grad(train_loss, param_influence)
            influences[train_guid] = torch.dot(
                inverse_hvp, gather_flat_grad(train_grads)
            ).item()

        break
    return influences


def compute_input_influence(
    full_model: nn.Module,
    test_guid: int,
    param_influence: List,
    train_dataset: Dataset,
    test_dataset: Dataset,
    training_indices=None,
    lissa_r: int = 1,
    lissa_depth: float = 0.25,
    damping=3e-3,
    scale=1e4,
):
    device = utils.get_device()

    train_dataloader_lissa = DataLoader(
        train_dataset, batch_size=16, shuffle=True, drop_last=True
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    for guid, input_ids, input_mask, label_ids in test_dataloader:
        if guid != test_guid:
            continue

        full_model.eval()

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        label_ids = label_ids.to(device)

        # L_TEST gradient
        full_model.zero_grad()
        output = full_model(input_ids, input_mask)
        test_loss = full_model.compute_loss(output, label_ids)
        test_grads = autograd.grad(test_loss, param_influence)

        # IVHP
        full_model.train()

        t = int(len(train_dataset) * lissa_depth)
        print(f"LiSSA reps: {lissa_r} and num_iterations: {t}")

        inverse_hvp = get_inverse_hvp_lissa(
            test_grads,
            full_model,
            device,
            param_influence,
            train_dataloader_lissa,
            damping=damping,
            scale=scale,
            num_samples=lissa_r,
            recursion_depth=t,
        )

        train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)

        _, input_id, _, _ = next(iter(train_dataloader))
        token_embed = full_model.bert.get_input_embeddings().weight[input_id].clone()
        num_tokens = token_embed.shape[1]
        embedding_size = token_embed.shape[2]

        influences = np.zeros((len(train_dataset), num_tokens, embedding_size))
        for train_guid, train_input_id, train_input_mask, train_label in tqdm(
            train_dataloader
        ):
            if training_indices is not None and train_guid not in training_indices:
                continue

            full_model.train()
            full_model.zero_grad()

            token_embeds = (
                full_model.bert.get_input_embeddings().weight[train_input_id].clone()
            )
            token_embeds.requires_grad = True

            train_output = full_model(
                inputs_embeds=token_embeds, attention_mask=train_input_mask
            )
            loss = full_model.compute_loss(train_output, train_label)

            grad_theta = torch.autograd.grad(loss, param_influence, create_graph=True)
            # grad_x = torch.autograd.grad(loss, token_embeds, create_graph=True)[0]

            flat_grad_theta = torch.cat([p.flatten() for p in grad_theta])
            num_params = len(flat_grad_theta)

            # compute the second-order partial derivative of the loss
            hessian = torch.zeros(
                (
                    num_params,
                    num_tokens,
                    embedding_size,
                )
            ).to(device)

            for p in range(num_params):
                grad2_x_theta = torch.autograd.grad(
                    flat_grad_theta[p], token_embeds, retain_graph=True
                )
                grad2_x_theta = torch.reshape(
                    grad2_x_theta[0], (num_tokens, embedding_size)
                )
                hessian[p] = grad2_x_theta

            # grad2_x_theta = torch.autograd.grad(flat_grad_theta, token_embeds)
            # for i in range(num_tokens):
            #     for j in range(embedding_size):
            #         grad2_x_theta = torch.autograd.grad(grad_x[0, i, j], param_influence)

            #         hessian[i, j] = torch.cat([grad2_x_theta[p_idx].flatten() for p_idx in range(num_params)])

            # # Reshape the gradient of the loss with respect to theta into a p by d matrix
            influences[train_guid] = torch.reshape(
                inverse_hvp @ hessian.view(num_params, -1), (num_tokens, embedding_size)
            ).cpu()
        break
    return influences
