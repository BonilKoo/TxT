import argparse

from datasets import *
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--embed_file', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--task_file') # multi-task

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--scaler', choices=[None, 'MinMax', 'Standard'], default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_time_intervals', type=int, default=64) # survival
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--xavier_uniform', action='store_true')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_epoch', type=int, default=100)

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
    
    parser.add_argument('--task', choices=['regression', 'classification', 'survival', 'multitask'])

    return parser.parse_args()