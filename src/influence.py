import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn


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


def hv(loss, model_params, v):  # according to pytorch issue #24004
    #     s = time.time()
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
    #     e1 = time.time()
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    #     e2 = time.time()
    #     print('1st back prop: {} sec. 2nd back prop: {} sec'.format(e1-s, e2-e1))
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
    for i in range(num_samples):
        cur_estimate = v
        lissa_data_iterator = iter(train_loader)
        for j in range(recursion_depth):
            try:
                input_ids, input_mask, segment_ids, label_ids, guids = next(
                    lissa_data_iterator
                )
            except StopIteration:
                lissa_data_iterator = iter(train_loader)
                input_ids, input_mask, segment_ids, label_ids, guids = next(
                    lissa_data_iterator
                )
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            model.zero_grad()
            train_loss = model(input_ids, segment_ids, input_mask, label_ids)
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


def compute_influence(model: nn.Module, train_dataloader, test_dataloader):


    
    pass
