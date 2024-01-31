import argparse
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch

from torch_geometric.nn import Node2Vec

from datasets import datasets
from utils import evaluation, utils

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--network_file', required=True, help='(csv) A network file representing relationships between genes.')
    parser.add_argument('--result_dir', required=True, help='(dir) A directory to save output files.')

    parser.add_argument('--seed', type=int, default=42, help='(int) Seed for random number generation, ensuring reproducibility of results. (default: 42)')
    parser.add_argument('--device', type=int, default=0, help='(int) Device number. (default: 0)')
    parser.add_argument('--val_ratio', type=float, default=0.05, help='(float) Ratio of data to use for validation. (default: 0.05)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='(float) Ratio of data to use for testing. (default: 0.1)')
    
    parser.add_argument('--batch_size', type=int, default=128, help='(int) Batch size for training, validation, and test sets. (default: 128)')
    parser.add_argument('--lr', type=float, default=0.01, help='(float) Learning rate for the optimizer. (default: 0.01)')
    parser.add_argument('--patience', type=int, default=10, help='(int) Number of epochs with no improvement in validation accuracy after which training will be stopped early. (default: 10)')
    parser.add_argument('--max_epoch', type=int, default=100, help='(int) Maximum number of training epochs. (default: 100)')

    parser.add_argument('--embedding_dim', type=int, default=64, help='(int) The size of each embedding vector. (default: 64)')
    parser.add_argument('--walk_length', type=int, default=20, help='(int) Length of the random walk per node. (default: 20)')
    parser.add_argument('--context_size', type=int, default=10, help='(int) Size of the context window. (default: 10)')
    parser.add_argument('--walks_per_node', type=int, default=10, help='(int) Number of random walks to start from each node. (default: 10)')
    parser.add_argument('--num_negative_samples', type=int, default=1, help='(int) Number of negative samples for each positive sample. (default: 1)')
    parser.add_argument('--p', type=float, default=1, help='(float) Likelihood of immediately revisiting a node in the walk. (default: 1)')
    parser.add_argument('--q', type=float, default=1, help='(float) Control parameter to interpolate between breadth-first strategy and depth-first strategy. (default: 1)')
    parser.add_argument('--sparse', action='store_true', help='An option to control the memory efficiency of storing random walks. (default: False)')

    return parser.parse_args()

def run(args):
    utils.set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    gene_list, train_data, val_data, test_data = datasets.load_network(args.network_file,
                                                                       args.val_ratio, args.test_ratio)
    
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

    node2vec_model_file = os.path.join(args.result_dir, 'node2vec_model.pt')
    clf_file = os.path.join(args.result_dir, 'link_prediction.joblib')
    log_file = os.path.join(args.result_dir, 'loss.csv')
    f = open(log_file, 'w')
    f.write('Epoch,Training Loss,Validation Accuracy,Validation AUROC\n')
    
    early_stopping = utils.EarlyStopping_node2vec(patience=args.patience, path_model=node2vec_model_file, path_clf=clf_file)

    for epoch in range(1, args.max_epoch+1):
        loss = train()
        clf, val_acc, val_auc = test()
        print(f'Epoch: {epoch:02d}, Training Loss: {loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUROC: {val_auc:.4f}')
        f.write(f'{epoch},{loss:.4f},{val_acc:.4f},{val_auc:.4f}\n')

        early_stopping(val_acc, model, clf)
        if early_stopping.early_stop:
            break
    f.close()

    model = utils.load_best_model(model, node2vec_model_file, device)
    clf = joblib.load(clf_file)
    evaluation.save_node2vec_result(model, [train_data, val_data, test_data], clf, args.result_dir)
    utils.save_embed(model, gene_list, args.result_dir)
#     rmfile(node2vec_model_file)
#     rmfile(clf_file)

def main():
    args = parse_args()
    utils.save_args(args)
    
    run(args)

if __name__ == '__main__':
    main()
