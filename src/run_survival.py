import argparse

import numpy as np

import torch
import torch.nn as nn

from datasets import SurvivalDataset, load_dataset
from evaluation import eval_result_survival
from models import TxT
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--embed_file', required=True)
    parser.add_argument('--result_dir', required=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--scaler', choices=[None, 'MinMax', 'Standard'], default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_time_intervals', type=int, default=64) # survival
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--xavier_uniform', action='store_true')
    parser.add_argument('--noam', action='store_true')
    parser.add_argument('--noam_factor', type=int, default=2)
    parser.add_argument('--noam_warmup', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--max_epoch', type=int, default=1000)
    
    parser.add_argument('--n_heads', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--norm_first', action='store_true')
    parser.add_argument('--n_layers', type=int, default=2)
    
    parser.add_argument('--aggfunc', choices=['Flatten', 'Avgpool'], default='Flatten')
    parser.add_argument('--d_hidden1', type=int, default=128)
    parser.add_argument('--d_hidden2', type=int, default=64)
    parser.add_argument('--slope', type=float, default=0.2)
    
    return parser.parse_args()

def run(args):
    set_seed(args.seed)
    
    train_dataloader, val_dataloader, test_dataloader, gene_list = load_dataset(args.input_file, args.output_file, args.val_ratio, args.test_ratio, args.scaler, args.batch_size, task='survival', n_time_intervals=args.n_time_intervals)
    num_times = train_dataloader.dataset.dataset.num_times
    time_buckets = train_dataloader.dataset.dataset.time_buckets
    times = train_dataloader.dataset.dataset.times
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    d_output_dict = {'task': num_times}

    model = CustomModel(args.embed_file, gene_list, device,
#                         n_heads, d_model, dropout, d_ff, norm_first, n_layers).to(device) # regression
                        args.n_heads, args.d_model, args.dropout, args.d_ff, args.norm_first, args.n_layers, # classification & survival
#                         args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, n_classes).to(device) # classification
                        args.aggfunc, args.d_hidden1, args.d_hidden2, args.slope, d_output_dict).to(device) # survival
    if args.xavier_uniform:
        for name, p in model.named_parameters():
            if ('embed' not in name) & (p.dim() > 1):
                nn.init.xavier_uniform_(p)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    Triangle = torch.FloatTensor(np.tri(num_times, num_times + 1, dtype=np.float32)).to(device) # survival
    
    def train():
        model.train()
        total_loss = 0
#         for x, y in train_dataloader: # regression & classification
        for x, y, T, E in train_dataloader: # survival
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
#             loss = criterion(output, y) # regression & classification
            loss = SurvivalLoss(output[0], y, E, Triangle) # survival
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(train_dataloader.dataset)
    
    @torch.no_grad()
    def test():
        model.eval()
        total_loss = 0
#         for x, y in val_dataloader: # regression & classification
        for x, y, T, E in val_dataloader: # survival
            x = x.to(device)
            y = y.to(device)
            
            output = model(x)
#             loss = criterion(output, y) # regression & classification
            loss = SurvivalLoss(output[0], y, E, Triangle) # survival
            
            total_loss += loss.item() * x.size(0)
        return total_loss / len(val_dataloader.dataset)
    
    model_file = f'{args.result_dir}/model.pt'
    log_file = f'{args.result_dir}/log.txt'
    f = open(log_file, 'w')
    early_stopping = EarlyStopping(patience=args.patience, path=model_file)

    for epoch in range(1, args.max_epoch+1):
        train_loss = train()
        
        if args.val_ratio > 0:
            val_loss = test()
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
            f.write(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}\n')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        else:
            print(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}')
            f.write(f'Epoch: {epoch:03d}, Traning Loss: {train_loss:.4f}\n')
        
        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f'Stop training because the training loss or validation loss is nan.')
            f.write(f'Stop training because the training loss or validation loss is nan.\n')
            break
    
    print(f'\nLog file is saved to {log_file}.')
    print(f'Model is saved to {model_file}.\n')
    f.write('\n')
    model = load_best_model(model, model_file, device)
    
    # regression
#     print_save_result(model, train_dataloader, device, f, 'Training')
#     print_save_result(model, val_dataloader, device, f, 'Validation')
#     print_save_result(model, test_dataloader, device, f, 'Test')
    
    # classification
#     print_save_result(model, train_dataloader, device, f, 'Training', n_classes)
#     print_save_result(model, val_dataloader, device, f, 'Validation', n_classes)
#     print_save_result(model, test_dataloader, device, f, 'Test', n_classes)
    
    # survival
    print_save_result(model, train_dataloader, device, f, 'Training', num_times, times)
    if args.val_ratio > 0:
        print_save_result(model, val_dataloader, device, f, 'Validation', num_times, times)
    print_save_result(model, test_dataloader, device, f, 'Test', num_times, times)
    
    f.close()

def main():
    args = parse_args()
    save_args(args)    
    run(args)

if __name__ == '__main__':
    main()
