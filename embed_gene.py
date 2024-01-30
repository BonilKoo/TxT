import argparse
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch

from torch_geometric.nn import Node2Vec

from evaluation import eval_result_node2vec
from datasets import load_network
from utils import *

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

    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--walk_length', type=int, default=20)
    parser.add_argument('--context_size', type=int, default=10)
    parser.add_argument('--walks_per_node', type=int, default=10)
    parser.add_argument('--num_negative_samples', type=int, default=1)
    parser.add_argument('--p', type=float, default=1)
    parser.add_argument('--q', type=float, default=1)
    parser.add_argument('--sparse', action='store_true')

    return parser.parse_args()

def run(args):
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    gene_list, train_data, val_data, test_data = load_network(args.network_file, args.val_ratio, args.test_ratio)
    
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
    
    early_stopping = EarlyStopping_node2vec(patience=args.patience, path_model=node2vec_model_file, path_clf=clf_file)

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
    test_acc, test_auc = eval_result_node2vec(model, test_data, clf)
    print(f'\nTest Accuracy: {test_acc:.4f}, Test AUROC: {test_auc:.4f}')
    f.write(f'\nTest Accuracy: {test_acc:.4f}, Test AUROC: {test_auc:.4f}\n')
    f.close()
    save_embed(model, gene_list, args.result_dir)
#     rmfile(node2vec_model_file)
#     rmfile(clf_file)

def main():
    args = parse_args()
    save_args(args)
    
    run(args)

if __name__ == '__main__':
    main()
