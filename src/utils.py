import copy
import joblib
import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

def mkdir(dir_name):
    os.makedirs(dir_name, exist_ok=True)

def rmfile(file_name):
    if os.path.exists(file_name):
        os.system(f'rm -f {file_name}')

def save_args(args):
    mkdir(args.result_dir)
    args_df = pd.DataFrame(vars(args), index=['value'])
    args_df.columns.name = 'argument'
    args_df.T.to_csv(f'{args.result_dir}/arguments.csv')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False

def load_best_model(model, best_model_file, device):
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    return model

def save_embed(model, gene_list, result_dir):
    pd.DataFrame(model.state_dict()['embedding.weight'].detach().cpu(), index=gene_list).to_csv(f'{result_dir}/embedding.csv')
    print(f'\nEmbedding is saved to {result_dir}/embedding.csv.\n')

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def SurvivalLoss(input, target, E, Triangle, reduction='mean'): # survival
    score_cens = input[E == 0]
    score_uncens = input[E == 1]
    target_cens = target[E == 0]
    target_uncens = target[E == 1]
    
    phi_uncens = torch.exp(torch.mm(score_uncens, Triangle))
    reduc_phi_uncens = torch.sum(phi_uncens * target_uncens, dim=1)
    
    phi_cens = torch.exp(torch.mm(score_cens, Triangle))
    reduc_phi_cens = torch.sum(phi_cens * target_cens, dim=1)
    
    z_uncens = torch.exp(torch.mm(score_uncens, Triangle))
    reduc_z_uncens = torch.sum(z_uncens, dim=1)
    
    z_cens = torch.exp(torch.mm(score_cens, Triangle))
    reduc_z_cens = torch.sum(z_cens, dim=1)
    
    loss = - (
                torch.sum(torch.log(reduc_phi_uncens)) \
                + torch.sum(torch.log(reduc_phi_cens)) \
                - torch.sum(torch.log(reduc_z_uncens)) \
                - torch.sum(torch.log(reduc_z_cens))
            )
    
    if reduction == 'mean':
        loss = loss / E.shape[0]
    
    return loss

def criterion(outputs, y_list, E, Triangle, task_name_dict, flag_survival): # naive (sum)
    criterion_regression = nn.MSELoss()
    criterion_classification = nn.CrossEntropyLoss()
    loss_list = []
    idx = 0
    if 'regression' in task_name_dict.keys():
        for name in task_name_dict['regression']:
            loss_list.append(criterion_regression(outputs[idx], y_list[idx]))
            idx += 1
    if 'classification' in task_name_dict.keys():
        for name in task_name_dict['classification']:
            loss_list.append(criterion_classification(outputs[idx], y_list[idx]))
            idx += 1
    if flag_survival:
        loss_list.append(SurvivalLoss(outputs[idx], y_list[idx], E, Triangle))

    return sum(loss_list)

class EarlyStopping_node2vec:
    """
    Code based on: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation accuracy doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path_model=None, path_clf=None, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str or None): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.path_model = path_model
        self.path_clf = path_clf
        self.trace_func = trace_func

    def __call__(self, val_acc, model, clf):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, clf)

        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, clf)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, clf):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation accuracy increased ({self.val_acc_min:.6f} --> {val_acc:.6f}).  Saving model ...')

        if self.path_model is not None:
            torch.save(model.state_dict(), self.path_model)
            joblib.dump(clf, self.path_clf)

        self.val_acc_max = val_acc

class EarlyStopping:
    """
    Code based on: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        if self.path is not None:
            torch.save(model.state_dict(), self.path)
        
        self.val_loss_min = val_loss