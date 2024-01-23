import argparse
import joblib
import os
import random

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch

from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.nn import Node2Vec

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network_file', required=True)
    parser.add_argument('--result_dir', required=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.05)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=100)
    
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--walk_length', type=int, default=20)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--num_negative_samples', type=int, default=1)
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    parser.add_argument('--sparse', action='store_true')
    
    return parser.parse_args()

def mkdir(dir_name):
    os.makedirs(dir_name, exit_ok=True)

def rmfile(file_name):
    if os.path.exists(file_name):
        os.system(f'rm -f {file_name}')

def save_args(args):
    mkdir(args.result_dir)
    args_df = pd.DataFrame(vars(args), index=['value'])
    args_df.columns.name = 'argument'
    args_df.T.to_csv(f'{args.result_dir}/arguments.tsv', sep='\t')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministric = True
    torch.backends.cudnn.benchmark = False

def load_network(network_file, val_ratio, test_ratio):
    network_df = pd.read_csv(network_file)
    print(f'\nNetwork file {network_file} is loaded.')
    
    network_nx = nx.from_pandas_edgelist(network_df, source=network_df.columns[0], target=network_df.columns[1])
    print(f'\n# of nodes = {len(network_nx)}')
    print(f'# of edges = {len(network_nx.edges())}\n')
    gene_list = network_nx.nodes()
    
    network_PyG = from_networkx(network_nx)
    
    random_link_split = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True)
    network_PyG = random_link_split(network_PyG)
    train_data = network_PyG[0]
    val_data = network_PyG[1]
    test_data = network_PyG[2]
    
    return gene_list, train_data, val_data, test_data

class EarlyStopping:
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

def load_best_model(model, best_model_file, device):
    model.load_state_dict(torch.load(best_model_file, map_location=device))
    return model

def eval_result(model, data, clf):
    model.eval()
    with torch.no_grad():
        z = model()
        z = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).detach().cpu().numpy()
        y = data.edge_label
        
        acc = clf.score(z, y)
        auc = roc_auc_score(y, clf.predict_proba(z)[:,1])
        return acc, auc

def save_result(model, gene_list, result_dir):
    pd.DataFrame(model.state_dict()['embedding.weight'].detach().cpu(), index=gene_list).to_csv(f'{result_dir}/embedding.csv')
    print(f'\nEmbedding is saved to {result_dir}/embedding.csv.')

def run(args):
    set_seed(args.seed)
    
    gene_list, train_data, val_data, test_data = load_network(args.network_file, args.val_ratio, args.test_ratio)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    model = Node2Vec(train_data.edge_index, embedding_dim=args.embedding_dim, walk_length=args.walk_length,
                     context_size=args.context_size, walks_per_node=args.walks_per_node,
                     num_negative_samples=args.num_negative_samples, p=args.p, q=args.q, sparse=args.sparse).to(device)
    
    loader = model.loader(batch_size=args.batch_size, shuffle=True)
    if args.sparse:
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr)
    
    def train():
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)
    
    @torch.no_grad()
    def test():
        model.eval()
        z = model()
        
        train_z = (z[train_data.edge_label_index[0]] * z[train_data.edge_label_index[1]]).detach().cpu().numpy()
        train_y = train_data.edge_label
        val_z = (z[val_data.edge_label_index[0]] * z[val_data.edge_label_index[1]]).detach().cpu().numpy()
        val_y = val_data.edge_label
        
        clf = LogisticRegression().fit(train_z, train_y)
        val_acc = clf.score(val_z, val_y)
        val_auc = roc_auc_score(val_y, clf.predict_proba(val_z)[:,1])
        return clf, val_acc, val_auc
    
    node2vec_model_file = f'{args.result_dir}/node2vec_model.pt'
    clf_file = f'{args.result_dir}/link_prediction.joblib'
    log_file = f'{args.result_dir}/log.txt'
    f = open(log_file, 'w')
    early_stopping = EarlyStopping(patience=args.patience, path_model=node2vec_model_file, path_clf=clf_file)
    
    for epoch in range(1, args.max_epoch+1):
        loss = train()
        clf, val_acc, val_auc = test()
        print(f'Epoch: {epoch:02d}, Training Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auc:.4f}')
        f.write(f'Epoch: {epoch:02d}, Training Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auc:.4f}\n')
        
        early_stopping(val_acc, model, clf)
        if early_stopping.early_stop:
            break
    
    model = load_best_model(model, node2vec_model_file, device)
    clf = joblib.load(clf_file)
    test_acc, test_auc = eval_result(model, test_data, clf)
    print(f'\nTest Accuracy: {test_acc:.4f}, Test AUROC: {test_auc:.4f}')
    f.write(f'\nTest Accuracy: {test_acc:.4f}, Test AUROC: {test_auc:.4f}\n')
    f.close()
    save_result(model, gene_list, args.result_dir)
    rmfile(node2vec_model_file)
    rmfile(clf_file)

def main():
    args = parse_args()
    save_args(args)
    run(args)

if __name__ == '__main__':
    main()