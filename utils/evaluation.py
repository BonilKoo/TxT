import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sksurv.metrics import concordance_index_censored, integrated_brier_score

import torch
import torch.nn.functional as F

def eval_node2vec(model, data, clf):
    model.eval()
    with torch.no_grad():
        z = model()
        z = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).detach().cpu().numpy()
        y = data.edge_label
        
        acc = clf.score(z, y)
        auc = roc_auc_score(y, clf.predict_proba(z)[:,1])
        return acc, auc

def save_node2vec_result(model, data, clf, result_dir):
    train_acc, train_auc = eval_node2vec(model, data[0], clf)
    val_acc, val_auc = eval_node2vec(model, data[1], clf)
    test_acc, test_auc = eval_node2vec(model, data[2], clf)
    with open(os.path.join(result_dir, 'performance.csv'), 'w') as f:
        f.write('Dataset,Accuracy,AUROC\n')
        f.write(f'Training,{train_acc:.4f},{train_auc:.4f}\n')
        f.write(f'Validation,{val_acc:.4f},{val_auc:.4f}\n')
        f.write(f'Test,{test_acc:.4f},{test_auc:.4f}\n')

def eval_result_regression(model, dataloader, device):
    y_true = np.empty(0)
    y_pred = torch.empty((0,1))
    
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            y_true = np.append(y_true, y)
            x = x.to(device)
            
            output = model(x)[0]
            
            y_pred = torch.cat((y_pred, output.detach().cpu()))
    
    y_true = torch.Tensor(y_true)
    y_pred = y_pred.squeeze(-1)
    
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    PCC = np.corrcoef(y_true, y_pred)[0][1]
    SCC = spearmanr(y_true, y_pred)[0]
    
    return MAE, RMSE, PCC, SCC

def eval_result_multitask_regression(model, dataloader, device, index): # regression
    y_true = np.empty(0)
    y_pred = torch.empty((0,1))

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])

            x = x.to(device)

            output = model(x)[index].detach().cpu()

            y_pred = torch.cat((y_pred, output))

    y_true = torch.Tensor(y_true)
    y_pred = y_pred.squeeze(-1)

    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    PCC = np.corrcoef(y_true, y_pred)[0][1]
    SCC = spearmanr(y_true, y_pred)[0]

    return MAE, RMSE, PCC, SCC

def print_save_result_regression(model, dataloader, device, dataset_type, result_dir):
    MAE, RMSE, PCC, SCC = eval_result_regression(model, dataloader, device)
    
    if not os.path.exists(f'{result_dir}/performance.csv'):
        log = open(f'{result_dir}/performance.csv', 'w')
        log.write(f'Dataset,MAE,RMSE,PCC,SCC\n')
    else:
        log = open(f'{result_dir}/performance.csv', 'a')
    
    log.write(f'{dataset_type},{MAE:.4f},{RMSE:.4f},{PCC:.4f},{SCC:.4f}\n')
    print(f'[{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}')
    log.close()

def eval_result_classification(model, dataloader, device): # classification
    n_classes = dataloader.dataset.dataset.d_output_dict['task']
    
    y_true = np.empty(0)
    y_score = torch.empty((0, n_classes))
    y_pred = np.empty(0)
    
    model.eval()
    with torch.no_grad():
        for x, y, _, _ in dataloader:
            y_true = np.append(y_true, y)
            
            x = x.to(device)
            
            output = model(x)[0].detach().cpu()
            
            y_score = torch.cat((y_score, output))
            output = F.softmax(output, dim=1).argmax(1)
            
            y_pred = np.append(y_pred, output.numpy())
        
    y_true = torch.IntTensor(y_true)
    y_score = F.softmax(y_score, dim=1)
    y_pred = torch.IntTensor(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_score[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')
    
    return accuracy, precision, recall, f1, auroc

def eval_result_multitask_classification(model, dataloader, device, n_classes, index): # classification
    y_true = np.empty(0)
    y_score = torch.empty((0, n_classes))
    y_pred = np.empty(0)

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])

            x = x.to(device)

            output = model(x)[index].detach().cpu()

            y_score = torch.cat((y_score, output))
            output = F.softmax(output, dim=1).argmax(1)

            y_pred = np.append(y_pred, output.numpy())

    y_true = torch.IntTensor(y_true)
    y_score = F.softmax(y_score, dim=1)
    y_pred = torch.IntTensor(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    if n_classes == 2:
        auroc = roc_auc_score(y_true, y_score[:, 1])
    else:
        auroc = roc_auc_score(y_true, y_score, average='macro', multi_class='ovr')

    return accuracy, precision, recall, f1, auroc

def print_save_result_classification(model, dataloader, device, dataset_type, result_dir):
    accuracy, precision, recall, f1, auroc = eval_result_classification(model, dataloader, device)
    
    if not os.path.exists(f'{result_dir}/performance.csv'):
        log = open(f'{result_dir}/performance.csv', 'w')
        log.write(f'Dataset,Accuracy,Precision,Recall,F1,AUROC\n')
    else:
        log = open(f'{result_dir}/performance.csv', 'a')
    
    log.write(f'{dataset_type},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{auroc:.4f}\n')
    print(f'[{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}')
    log.close()

def predict(score, num_times): # survival
    Triangle1 = np.tri(num_times, num_times + 1)
    Triangle2 = np.tri(num_times + 1, num_times + 1)
    
    phi = np.exp(np.dot(score, Triangle1))
    div = np.repeat(np.sum(phi, 1).reshape(-1, 1), phi.shape[1], axis=1)
    density = phi / div
    Survival = np.dot(density, Triangle2)
    hazard = density[:, :-1] / Survival[:, 1:]
    
    return hazard, density, Survival

def predict_hazard(score, num_times): # survival
    hazard, _, _ = predict(score, num_times)
    
    return hazard

def predict_survival(score, num_times): # survival
    _, _, survival = predict(score, num_times)
    return survival

def predict_cumulative_hazard(score, num_times): # survival
    hazard = predict_hazard(score, num_times)
    cumulative_hazard = np.cumsum(hazard, 1)
    return cumulative_hazard

def predict_risk(score, num_times, use_log=False): # survival
    cumulative_hazard = predict_cumulative_hazard(score, num_times)
    risk_score = np.sum(cumulative_hazard, 1)
    if use_log:
        return np.log(risk_score)
    else:
        return risk_score

def c_index(score, T, E, num_times): # survival
    risk = predict_risk(score, num_times)
    
    result = concordance_index_censored(E.astype(bool), T, risk)[0]
    
    return result

def ibs(score, T, E, num_times, times): # survival
    Survival = predict_survival(score, num_times)
    
    E_bool = E.astype(bool)
    true = np.array([(E_bool[i], T[i]) for i in range(len(E))], dtype=[('event', np.bool_), ('time', np.float32)])
    
    max_time = max(T)
    min_time = min(T)
    
    valid_index = [i for i in range(len(times)) if min_time <= times[i] <= max_time]
    times = times[valid_index]
    Survival = Survival[:, valid_index]
    
    result = integrated_brier_score(true, true, Survival, times)
    
    return result

def eval_result_survival(model, dataloader, device, num_times, times): # survival
    y_true = np.empty((0, num_times + 1))
    T_true = np.empty(0)
    E_true = np.empty(0)
    score = np.empty((0, num_times))
    
    model.eval()
    with torch.no_grad():
        for x, y, T, E in dataloader:
            y_true = np.append(y_true, y)
            T_true = np.append(T_true, T)
            E_true = np.append(E_true, E)
            
            x = x.to(device)
            
            output = model(x)[0].detach().cpu().numpy()
            
            score = np.append(score, output, axis=0)
    
    C_Index = c_index(score, T_true, E_true, num_times)
    IBS = ibs(score, T_true, E_true, num_times, times)
    
    return C_Index, IBS

def eval_result_multitask_survival(model, dataloader, device, num_times, times, index):
    y_true = np.empty((0, num_times + 1))
    T_true = np.empty(0)
    E_true = np.empty(0)
    score = np.empty((0, num_times))

    model.eval()
    with torch.no_grad():
        for x, *y_list, T, E in dataloader:
            y_true = np.append(y_true, y_list[index])
            T_true = np.append(T_true, T)
            E_true = np.append(E_true, E)

            x = x.to(device)

            output = model(x)[index].detach().cpu().numpy()

            score = np.append(score, output, axis=0)

    C_Index = c_index(score, T_true, E_true, num_times)
    IBS = ibs(score, T_true, E_true, num_times, times)

    return C_Index, IBS
    
def print_save_result_survival(model, dataloader, device, dataset_type, result_dir, num_times, times):
    C_Index, IBS = eval_result_survival(model, dataloader, device, num_times, times)
    
    if not os.path.exists(f'{result_dir}/performance.csv'):
        log = open(f'{result_dir}/performance.csv', 'w')
        log.write(f'Dataset,C-Index,IBS\n')
    else:
        log = open(f'{result_dir}/performance.csv', 'a')
    
    log.write(f'{dataset_type},{C_Index:.4f},{IBS:.4f}\n')
    print(f'[{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}')
    log.close()

def print_save_result_multitask(model, dataloaders, device, result_dir, val_ratio,
                      task_name_dict, d_output_dict, num_times, times, flag_survival):
    dataset_types = ['Training', 'Validation', 'Test']    
    for dataloader, dataset_type in zip(dataloaders, dataset_types):
        if dataset_type == 'Validation' and val_ratio == 0:
            continue
        
        idx = 0
        if 'regression' in task_name_dict.keys():
            for name in task_name_dict['regression']:
                if not os.path.exists(f'{result_dir}/performance_{name}.csv'):
                    log = open(f'{result_dir}/performance_{name}.csv', 'w')
                    log.write('Dataset,MAE,RMSE,PCC,SCC\n')
                else:
                    log = open(f'{result_dir}/performance_{name}.csv', 'a')
                MAE, RMSE, PCC, SCC = eval_result_multitask_regression(model, dataloader, device, idx)
                log.write(f'{dataset_type},{MAE:.4f},{RMSE:.4f},{PCC:.4f},{SCC:.4f}\n')
                print(f'[{name:^17s}] [{dataset_type:^10s}] MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, PCC: {PCC:.4f}, SCC: {SCC:.4f}')
                log.close()
                idx += 1

        if 'classification' in task_name_dict.keys():
            for name in task_name_dict['classification']:
                if not os.path.exists(f'{result_dir}/performance_{name}.csv'):
                    log = open(f'{result_dir}/performance_{name}.csv', 'w')
                    log.write('Dataset,Accuracy,Precision,Recall,F1,AUROC\n')
                else:
                    log = open(f'{result_dir}/performance_{name}.csv', 'a')
                n_classes = d_output_dict[name]
                accuracy, precision, recall, f1, auroc = eval_result_multitask_classification(model, dataloader, device, n_classes, idx) # classification
                log.write(f'{dataset_type},{accuracy:.4f},{precision:.4f},{recall:.4f},{f1:.4f},{auroc:.4f}\n') # classification
                print(f'[{name:^17s}] [{dataset_type:^10s}] Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}') # classification
                log.close()
                idx += 1

        if flag_survival:
            name = 'survival'
            if not os.path.exists(f'{result_dir}/performance_{name}.csv'):
                log = open(f'{result_dir}/performance_{name}.csv', 'w')
                log.write('Dataset,C-Index,IBS\n')
            else:
                log = open(f'{result_dir}/performance_{name}.csv', 'a')
            C_Index, IBS = eval_result_multitask_survival(model, dataloader, device, num_times, times, idx) # survival
            log.write(f'{dataset_type},{C_Index:.4f},{IBS:.4f}\n') # survival
            print(f'[{name:^17s}] [{dataset_type:^10s}] C-Index: {C_Index:.4f}, IBS: {IBS:.4f}') # survival
            log.close()

        print('\n')

def print_save_result(model, dataloaders, device, result_dir, task, val_ratio, num_times, times):
    if task == 'regression':
        print_save_result_regression(model=model, dataloader=dataloaders[0], device=device, dataset_type='Training', result_dir=result_dir)
        if val_ratio > 0:
            print_save_result_regression(model=model, dataloader=dataloaders[1], device=device, dataset_type='Validation', result_dir=result_dir)
        print_save_result_regression(model=model, dataloader=dataloaders[2], device=device, dataset_type='Test', result_dir=result_dir)
    
    elif task == 'classification':
        print_save_result_classification(model=model, dataloader=dataloaders[0], device=device, dataset_type='Training', result_dir=result_dir)
        if val_ratio > 0:
            print_save_result_classification(model=model, dataloader=dataloaders[1], device=device, dataset_type='Validation', result_dir=result_dir)
        print_save_result_classification(model=model, dataloader=dataloaders[2], device=device, dataset_type='Test', result_dir=result_dir)
    
    elif task == 'survival':
        print_save_result_survival(model=model, dataloader=dataloaders[0], device=device, dataset_type='Training', result_dir=result_dir, num_times=num_times, times=times)
        if val_ratio > 0:
            print_save_result_survival(model=model, dataloader=dataloaders[1], device=device, dataset_type='Validation', result_dir=result_dir, num_times=num_times, times=times)
        print_save_result_survival(model=model, dataloader=dataloaders[2], device=device, dataset_type='Test', result_dir=result_dir, num_times=num_times, times=times)
    
    else:
        raise NotImplementedError