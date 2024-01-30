from itertools import accumulate
import random

def PCGrad_backward(optimizer, outputs, y_list, n_tasks, idx_task_dict, flag_survival, E, Triangle, device): # PCGrad
    '''Code based on: https://github.com/wgchang/PCGrad-pytorch-example/blob/master/pcgrad-example.py'''
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    
    grads_task = []
    grad_shapes = [p.shape if p.requires_grad is True else None
                   for group in optimizer.param_groups for p in group['params']]
    grad_numel = [p.numel() if p.requires_grad is True else 0
                  for group in optimizer.param_groups for p in group['params']] # total number of elements
    
    loss_list = []
    optimizer.zero_grad()
    
    # calculate gradients for each task
    for idx in range(n_tasks):
        if idx_task_dict[idx] == 'regression':
            loss = criterion_regression(outputs[idx], y_list[idx])
        elif idx_task_dict[idx] == 'classification':
            loss = criterion_classification(outputs[idx], y_list[idx])
        else:
            loss = SurvivalLoss(outputs[idx], y_list[idx], E, Triangle)
        loss_list.append(loss)
        # 1
        loss.backward(retain_graph=True)

        grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in optimizer.param_groups for p in group['params']]

        # fill zero grad if grad is None but requires_grad is true
        grads_task.append(torch.cat([g if g is not None else torch.zeros(
            grad_numel[i], device=device) for i, g in enumerate(grad)]))
        optimizer.zero_grad()
        
    # shuffle gradient order
    # 3 & 4
    random.shuffle(grads_task)

    # gradient projection
    grads_task = torch.stack(grads_task, dim=0)  # (T, # of params)
#     grads_task = torch.stack(grads_task, dim=0).detach().cpu()  # (T, # of params)
    # 2
    proj_grad = grads_task.clone()

    # 5 & 6 & 7
    def _proj_grad(grad_task):
        for k in range(n_tasks):
            inner_product = torch.sum(grad_task * grads_task[k])
            proj_direction = inner_product / (torch.sum(grads_task[k] * grads_task[k]) + 1e-12)
            grad_task = grad_task - torch.min(proj_direction, torch.zeros_like(proj_direction)) * grads_task[k]
        return grad_task

    proj_grad = torch.sum(torch.stack(list(map(_proj_grad, list(proj_grad)))), dim=0)  # (of params, )

    indices = [0, ] + [v for v in accumulate(grad_numel)]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(grad_shapes) == len(indices[:-1])
    for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
        if grad_shape is not None:
            param.grad[...] = proj_grad[start_idx:end_idx].view(grad_shape)  # copy proj grad

    return loss_list