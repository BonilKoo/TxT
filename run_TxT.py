import argparse

import numpy as np

import torch
import torch.nn as nn

from datasets.datasets import *
from models.models import TxT
from utils.evaluation import *
from utils.pcgrad import PCGrad_backward
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', required=True, help='(csv) A omics profile file representing a gene expression dataset where each row corresponds to a sample, and each column, labeled with gene names, represents the expression level of the corresponding gene in the respective sample. The numerical values in the matrix indicate the expression levels of each gene in the corresponding samples.')
    parser.add_argument('--output_file', required=True, help='(tsv) A file containing clinical feature data. The format is organized with a header line indicating the type of data and subsequent rows containing sample-specific information.')
    parser.add_argument('--embed_file', required=True, help='(csv) A csv file representing gene embedding. The gene names are listed in the first column, and the subsequent columns contain the embedding values for each gene in different dimensions.')
    parser.add_argument('--result_dir', required=True, help='(dir) A directory to save output files.')
    parser.add_argument('--task', required=True, choices=['regression', 'classification', 'survival', 'multitask'], help='(str) The type of task to perform. Choose among [regression/classification/survival/multitask].')
    parser.add_argument('--task_file', default=None, help='(tsv) Only for multi-task learning. A tsv file that outlines a set of tasks. Each row in the file represents a specific task, and the information is organized into two columns: "task name" and "prediction type".')

    parser.add_argument('--seed', type=int, default=42, help='(int) Seed for random number generation, ensuring reproducibility of results. (default: 42)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='(float) Ratio of data to use for validation. (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='(float) Ratio of data to use for testing. (default: 0.2)')
    parser.add_argument('--scaler', choices=[None, 'MinMax', 'Standard'], default='MinMax', help='(str) A data normalization method. Choose among [MinMax/Standard/None]. The gene expression levels were normalized by using the expression values of the traning set. (default: MinMax)')
    parser.add_argument('--batch_size', type=int, default=64, help='(int) Batch size for training, validation, and test sets. (default: 64)')
    parser.add_argument('--n_time_intervals', type=int, default=64, help='(int) Number of time intervals for survival prediction. (default: 64)') # survival
    
    parser.add_argument('--device', type=int, default=0, help='(int) Device number.')
    parser.add_argument('--xavier_uniform', action='store_true', help='An option to use Xavier Uniform initialization for model weights to prevent issues like vanishing or exploding gradients during the training process.')
    parser.add_argument('--lr', type=float, default=0.0001, help='(float) Learning rate for the optimizer. (default: 0.0001)')
    parser.add_argument('--patience', type=int, default=50, help='(int) Number of epochs with no improvement in validation loss after which training will be stopped early. (default: 50)')
    parser.add_argument('--max_epoch', type=int, default=1000, help='(int) Maximum number of training epochs. (default: 1000)')

    parser.add_argument('--n_heads', type=int, default=2, help='(int) Number of attention heads. (default: 2)')
    parser.add_argument('--d_model', type=int, default=64, help='(int) Dimensionality of transformer. (default: 64)')
    parser.add_argument('--dropout', type=float, default=0.1, help='(float) Dropout rate. (default: 0.1)')
    parser.add_argument('--d_ff', type=int, default=256, help='(int) Dimensionality of the feed-forward layer. (default: 256)')
    parser.add_argument('--norm_first', action='store_true', help='An option to perform LayerNorms before other attention and feedforward operations, otherwise after. (default: False)')
    parser.add_argument('--n_layers', type=int, default=2, help='(int) Number of layers in transformer. (default: 2)')

    parser.add_argument('--aggfunc', choices=['Flatten', 'Avgpool'], default='Flatten', help='(str) Aggregation function after transformer module. Choose between [Flatten/Avgpool]. (default: Flatten)')
    parser.add_argument('--d_hidden1', type=int, default=128, help='(int) Dimensionality of the first hidden layer in task-specific layer. (default: 128)')
    parser.add_argument('--d_hidden2', type=int, default=64, help='(int) Dimensionality of the second hidden layer in task-specific layer. (default: 65)')
    parser.add_argument('--slope', type=float, default=0.2, help='(float) Slope parameter for the Leaky ReLU activation function. (default: 0.2)')

    return parser.parse_args()

def run(args):
    set_seed(seed=args.seed)
    
    train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(input_file=args.input_file,
                                                                                output_file=args.output_file,
                                                                                task=args.task,
                                                                                val_ratio=args.val_ratio,
                                                                                test_ratio=args.test_ratio,
                                                                                scaler=args.scaler,
                                                                                batch_size=args.batch_size,
                                                                                n_time_intervals=args.n_time_intervals,
                                                                                task_file=args.task_file)
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    d_output_dict = train_dataloader.dataset.dataset.d_output_dict
    if args.task == 'multitask':
        task_name_dict = train_dataloader.dataset.dataset.task_name_dict
        flag_survival = train_dataloader.dataset.dataset.flag_survival
        n_tasks = train_dataloader.dataset.dataset.n_tasks # PCGrad
        idx_task_dict = train_dataloader.dataset.dataset.idx_task_dict # PCGrad
    elif args.task == 'survival':
        flag_survival = 1
    else:
        flag_survival = 0
    
    model = TxT(args.embed_file, gene_list, device,
                        args.n_heads, args.d_model, args.dropout, args.d_ff, args.norm_first, args.n_layers,
                        args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, d_output_dict).to(device)
    if args.xavier_uniform:
        for name, p in model.named_parameters():
            if ('embed' not in name) & (p.dim() > 1):
                nn.init.xavier_uniform_(p)
    
    if args.task == 'survival' or flag_survival:
        num_times = train_dataloader.dataset.dataset.num_times
#         time_buckets = train_dataloader.dataset.dataset.time_buckets
        times = train_dataloader.dataset.dataset.times
    
        Triangle = torch.FloatTensor(np.tri(num_times, num_times + 1, dtype=np.float32)).to(device) # survival
    
    else:
        num_times = None
        times = None
        Triangle = None
    
    if args.task == 'regression':
        criterion = nn.MSELoss()
    elif args.task == 'classification':
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    def train():
        model.train()
        total_loss = 0
#         for x, y, T, E in train_dataloader:
        for x, *y_list, T, E in train_dataloader:
            x = x.to(device)
#             y = y.to(device)
            y_list = [y.to(device) for y in y_list]
            
#             output = model(x)[0]
            outputs = model(x)
            if args.task in ['regression', 'classification']:
#                 loss = criterion(output, y)
                loss = criterion(outputs[0], y_list[0])
            elif args.task == 'survival':
#                 loss = SurvivalLoss(output, y, E, Triangle) # survival
                loss = SurvivalLoss(outputs[0], y_list[0], E, Triangle) # survival
            elif args.task == 'multitask':
                loss_list = PCGrad_backward(optimizer, outputs, y_list, n_tasks, idx_task_dict, flag_survival, E, Triangle, device, False) # PCGrad
                loss = sum(loss_list)
            else:
                raise NotImplementedError
            
            if args.task != 'multitask':
                optimizer.zero_grad()
                loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_dataloader.dataset)
    
    @torch.no_grad()
    def test():
        model.eval()
        total_loss = 0
#         for x, y, T, E in val_dataloader:
        for x, *y_list, T, E in val_dataloader:
            x = x.to(device)
#             y = y.to(device)
            y_list = [y.to(device) for y in y_list]
            
#             output = model(x)[0]
            outputs = model(x)
            if args.task in ['regression', 'classification']:
#                 loss = criterion(output, y)
                loss = criterion(outputs[0], y_list[0])
            elif args.task == 'survival':
#                 loss = SurvivalLoss(output, y, E, Triangle) # survival
                loss = SurvivalLoss(outputs[0], y_list[0], E, Triangle) # survival
            elif args.task == 'multitask':
                loss_list = PCGrad_backward(optimizer, outputs, y_list, n_tasks, idx_task_dict, flag_survival, E, Triangle, device, True) # PCGrad
                loss = sum(loss_list)
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(val_dataloader.dataset)
    
    model_file = f'{args.result_dir}/TxT.pt'
    loss_file = f'{args.result_dir}/loss.csv'
    f = open(loss_file, 'w')
    if args.val_ratio > 0:
        f.write('Epoch,Training Loss,Validation Loss\n')
    else:
        f.write('Epoch,Training Loss\n')
    early_stopping = EarlyStopping(patience=args.patience, path=model_file)

    for epoch in range(1, args.max_epoch+1):
        train_loss = train()
        
        if args.val_ratio > 0:
            val_loss = test()
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            f.write(f'{epoch},{train_loss:.4f},{val_loss:.4f}\n')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        else:
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}')
            f.write(f'{epoch:03d},{train_loss:.4f}\n')
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f'Stop training because the training loss or validation loss is nan.')
            break
    
    print(f'\nLoss is saved to {loss_file}.')
    f.close()
    print(f'Model is saved to {model_file}.\n')
    model = load_best_model(model, model_file, device)
    
    if args.task != 'multitask':
        print_save_result(model=model, dataloaders=[train_dataloader,val_dataloader,test_dataloader], device=device, result_dir=args.result_dir, task=args.task, val_ratio=args.val_ratio, num_times=num_times, times=times)
    else:
        print_save_result_multitask(model=model, dataloaders=[train_dataloader,val_dataloader,test_dataloader], device=device, result_dir=args.result_dir, val_ratio=args.val_ratio, task_name_dict=task_name_dict, d_output_dict=d_output_dict, num_times=num_times, times=times, flag_survival=flag_survival)

def main():
    args = parse_args()
    check_args(args)
    save_args(args)
    
    run(args)

if __name__ == '__main__':
    main()

