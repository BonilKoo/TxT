from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
from torch.utils.data import Dataset, random_split, DataLoader

from torch_geometric.utils.convert import from_networkx
from torch_geometric.transforms import RandomLinkSplit

def load_network(network_file, val_ratio, test_ratio):
    network_df = pd.read_csv(network_file)
    print(f'\nNetwork file {network_file} is loaded.\n')
    
    network_nx = nx.from_pandas_edgelist(network_df, source=network_df.columns[0], target=network_df.columns[1])
    print(f'# of nodes = {len(network_nx)}')
    print(f'# of edges = {len(network_nx.edges())}\n')
    gene_list = network_nx.nodes()
    
    network_PyG = from_networkx(network_nx)
    
    random_link_split = RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True)
    train_data, val_data, test_data = random_link_split(network_PyG)
    
    return gene_list, train_data, val_data, test_data

class RegressionDataset(Dataset):
    def __init__(self, input_file, output_file):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_csv(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')
        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}\n')
        
        self.gene_list = input_df.columns.to_list()
        
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
        
        self.length = len(df)
        
        self.d_output_dict = {'task': 1}
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = [torch.FloatTensor([self.y[index]])]
        return x, *y, [0], [0]
    
    def __len__(self):
        return self.length

class ClassificationDataset(Dataset):
    def __init__(self, input_file, output_file):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_csv(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')
        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}\n')
        
        self.gene_list = input_df.columns.to_list()
        
        self.x = df.iloc[:, :-1].values
        self.y = df.iloc[:, -1].values
        
        self.label_dict = {label: idx for idx, label in enumerate(np.unique(self.y))}
        self.n_classes = len(self.label_dict)
        self.y = list(map(self.label_to_vector, self.y))
        
        self.length = len(df)
        
        self.d_output_dict = {'task': self.n_classes}
    
    def label_to_vector(self, value):
        return self.label_dict.get(value, None)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = [torch.LongTensor(self.y)[index]]
        return x, *y, [0], [0]
    
    def __len__(self):
        return self.length

class SurvivalDataset(Dataset):
    def __init__(self, input_file, output_file, n_time_intervals, is_min_time_zero=True, extra_pct_time=0.1):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_csv(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')
        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}\n')
        
        self.gene_list = input_df.columns.to_list()
        
        self.x = df.iloc[:, :-2].values
        self.T = df.iloc[:, -2].values
        self.E = df.iloc[:, -1].values
        
        self.n_time_intervals = n_time_intervals
        self.y = self.compute_Y(self.T, self.E, is_min_time_zero, extra_pct_time)
        
        self.length = len(df)
        
        self.d_output_dict = {'task': self.num_times}
    
    def label_to_vector(self, value):
        return self.label_dict.get(value, None)
    
    def get_time_buckets(self):
        return [(self.times[i], self.times[i+1]) for i in range(len(self.times) - 1)]
    
    def get_times(self, T, is_min_time_zero, extra_pct_time):
        max_time = max(T)
        if is_min_time_zero:
            min_time = 0
        else:
            min_time = min(T)
        
        if 0 <= extra_pct_time <= 1:
            p = extra_pct_time
        else:
            raise Exception('"extra_pct_time" has to be between [0,1].')
        
        self.times = np.linspace(min_time, max_time * (1 + p), self.n_time_intervals)
        self.time_buckets = self.get_time_buckets()
        self.num_times = len(self.time_buckets)
    
    def compute_Y(self, T, E, is_min_time_zero, extra_pct_time):
        self.get_times(T, is_min_time_zero, extra_pct_time)
        
        Y = []
        
        for t, e in zip(T, E):
            y = np.zeros(self.num_times + 1)
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)
            
            if e == 1:
                y[index] = 1
                Y.append(y.tolist())
            else:
                y[index:] = 1
                Y.append(y.tolist())
        
        return torch.FloatTensor(Y)
    
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y = [torch.FloatTensor(self.y[index])]
        T = torch.FloatTensor(self.T)[index]
        E = torch.LongTensor(self.E)[index]
        return x, *y, T, E
    
    def __len__(self):
        return self.length

class MultitaskDataset(Dataset):
    def __init__(self, input_file, output_file, task_file, n_time_intervals, is_min_time_zero=True, extra_pct_time=0.1):
        input_df = pd.read_csv(input_file, index_col=0)
        print(f'\nInput file {input_file} is loaded.')
        output_df = pd.read_csv(output_file, index_col=0)
        print(f'Output file {output_file} is loaded.')

        task_df = pd.read_csv(task_file)
        print(f'Task file {task_file} is loaded.')
        task_df_regression = task_df[task_df['task'] == 'regression']
        task_df_classification = task_df[task_df['task'] == 'classification']
        task_df_survival = task_df[task_df['task'].str.contains('survival')]
        task_df = pd.concat([task_df_regression, task_df_classification, task_df_survival], ignore_index=True)
        output_df = output_df[task_df['name']]

        df = pd.merge(input_df, output_df, left_index=True, right_index=True)
        print(f'\n# of samples = {len(df)}')
        print(f'# of genes = {df.shape[1] - output_df.shape[1]}')


        if ('survival_time' in task_df['task'].to_list()) and ('survival_event' in task_df['task'].to_list()):
            self.flag_survival = 1
        elif ('survival_time' not in task_df['task'].to_list()) and ('survival_event' not in task_df['task'].to_list()):
            self.flag_survival = 0
        else:
            raise Exception('"survival_time" and "survival_event" must be included together.')
        self.n_tasks = task_df.shape[0] - self.flag_survival
        print(f'# of tasks = {self.n_tasks}\n')

        self.name_task_dict = OrderedDict({name: task for name, task in task_df.values})

        self.task_name_dict = OrderedDict()
        for name, task in task_df.values:
            if task in self.task_name_dict.keys():
                self.task_name_dict[task].append(name)
            else:
                if 'survival' in task:
                    self.task_name_dict[task] = name
                else:
                    self.task_name_dict[task] = [name]
        
        # PCGrad
        if 'regression' in self.task_name_dict.keys():
            flag_regression = True
        else:
            flag_regression = False
        if 'classification' in self.task_name_dict.keys():
            flag_classification = True
        else:
            flag_classification = False
        
        self.idx_task_dict = OrderedDict()
        if flag_regression:
            for i in range(len(self.task_name_dict['regression'])):
                self.idx_task_dict[len(self.idx_task_dict)] = 'regression'
        if flag_classification:
            for i in range(len(self.task_name_dict['classification'])):
                self.idx_task_dict[len(self.idx_task_dict)] = 'classification'
        if self.flag_survival:
            self.idx_task_dict[len(self.idx_task_dict)] = 'survival'
        # PCGrad

        self.gene_list = input_df.columns.to_list()

        self.x = df.iloc[:, :-output_df.shape[1]].values

        self.y_df = df.iloc[:, -output_df.shape[1]:]
        self.classification_name_label_dict = OrderedDict()
        self.d_output_dict = OrderedDict()
        for task in self.task_name_dict.keys():
            if task == 'regression':
                for name in self.task_name_dict[task]:
                    self.d_output_dict[name] = 1
            elif task == 'classification':
                for name in self.task_name_dict[task]:
                    self.classification_name_label_dict[name] = {label: idx for idx, label in enumerate(np.unique(self.y_df[name]))}
                    self.tmp_dict = self.classification_name_label_dict[name]
                    self.y_df[name] = list(map(self.label_to_vector, self.y_df[name]))
                    del self.tmp_dict
                    self.d_output_dict[name] = len(self.classification_name_label_dict[name])
            elif task == 'survival_time':
                self.T = self.y_df[self.task_name_dict['survival_time']].values
                self.E = self.y_df[self.task_name_dict['survival_event']].values
                self.y_df.drop(columns=[self.task_name_dict['survival_time'], self.task_name_dict['survival_event']], inplace=True)
                self.n_time_intervals = n_time_intervals
                self.y_survival = self.compute_Y_survival(self.T, self.E, is_min_time_zero, extra_pct_time)
                self.d_output_dict['survival_time'] = self.num_times
            else:
                continue

        self.length = len(df)

    def label_to_vector(self, value):
        return self.tmp_dict.get(value, None)

    def get_time_buckets(self):
        return [(self.times[i], self.times[i+1]) for i in range(len(self.times) - 1)]

    def get_times(self, T, is_min_time_zero, extra_pct_time): # survival
        max_time = max(T)
        if is_min_time_zero:
            min_time = 0
        else:
            min_time = min(T)

        if 0 <= extra_pct_time <= 1:
            p = extra_pct_time
        else:
            raise Exception('"extra_pct_time" has to be between [0,1].')

        self.times = np.linspace(min_time, max_time * (1 + p), self.n_time_intervals)
        self.time_buckets = self.get_time_buckets()
        self.num_times = len(self.time_buckets)

    def compute_Y_survival(self, T, E, is_min_time_zero, extra_pct_time):
        self.get_times(T, is_min_time_zero, extra_pct_time)

        Y = []

        for t, e in zip(T, E):
            y = np.zeros(self.num_times + 1)
            min_abs_value = [abs(a_j_1 - t) for (a_j_1, a_j) in self.time_buckets]
            index = np.argmin(min_abs_value)

            if e == 1:
                y[index] = 1
                Y.append(y.tolist())
            else:
                y[index:] = 1
                Y.append(y.tolist())

        return torch.FloatTensor(Y)

    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index])
        y_list = []
        for task in self.task_name_dict.keys():
            if task == 'regression':
                for name in self.task_name_dict[task]:
                    y_list.append(torch.FloatTensor([self.y_df[name].values[index]]))
            elif task == 'classification':
                for name in self.task_name_dict[task]:
                    y_list.append(torch.LongTensor(self.y_df[name].values)[index])
        if self.flag_survival:
            y_list.append(torch.FloatTensor(self.y_survival[index]))
            T = torch.FloatTensor(self.T)[index]
            E = torch.FloatTensor(self.E)[index]
        else:
            T = []
            E = []
        return x, *y_list, T, E

    def __len__(self):
        return self.length

def load_dataset(input_file, output_file, task, val_ratio=0.1, test_ratio=0.2, scaler='MinMax', batch_size=64, n_time_intervals=64, task_file=None):
    if task == 'regression':
        dataset = RegressionDataset(input_file, output_file)
    elif task == 'classification':
        dataset = ClassificationDataset(input_file, output_file)
    elif task == 'survival':
        dataset = SurvivalDataset(input_file, output_file, n_time_intervals)
    elif task == 'multitask':
        dataset = MultitaskDataset(input_file, output_file, task_file, n_time_intervals)
    else:
        raise
    
    dataset_size = len(dataset)
    train_ratio = 1 - val_ratio - test_ratio
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    elif scaler == 'Standard':
        scaler = StandardScaler()
    
    if scaler != 'None':
        train_X_scaled = scaler.fit_transform(dataset.x[train_dataset.indices])
        val_X_scaled = scaler.transform(dataset.x[val_dataset.indices])
        test_X_scaled = scaler.transform(dataset.x[test_dataset.indices])

        dataset.x[train_dataset.indices] = train_X_scaled
        dataset.x[val_dataset.indices] = val_X_scaled
        dataset.x[test_dataset.indices] = test_X_scaled
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    gene_list = dataset.gene_list
    
    return train_dataloader, val_dataloader, test_dataloader, gene_list