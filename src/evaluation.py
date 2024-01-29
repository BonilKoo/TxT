from sklearn.metrics import roc_auc_score

import torch

def eval_result_node2vec(model, data, clf):
    model.eval()
    with torch.no_grad():
        z = model()
        z = (z[data.edge_label_index[0]] * z[data.edge_label_index[1]]).detach().cpu().numpy()
        y = data.edge_label
        
        acc = clf.score(z, y)
        auc = roc_auc_score(y, clf.predict_proba(z)[:,1])
        return acc, auc

def eval_result_regression(model, dataloader, device):
    y_true = np.empty(0)
    y_pred = torch.empty((0,1))
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_true = np.append(y_true, y)
            x = x.to(device)
            
            output = model(x)
            
            y_pred = torch.cat((y_pred, output.detach().cpu()))
    
    y_true = torch.Tensor(y_true)
    y_pred = y_pred.squeeze(-1)
    
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    PCC = np.corrcoef(y_true, y_pred)[0][1]
    SCC = spearmanr(y_true, y_pred)[0]
    
    return MAE, RMSE, PCC, SCC

def eval_result_classification(model, dataloader, device, n_classes): # classification
    y_true = np.empty(0)
    y_score = torch.empty((0, n_classes))
    y_pred = np.empty(0)
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            y_true = np.append(y_true, y)
            
            x = x.to(device)
            
            output = model(x).detach().cpu()
            
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
            
            output = model(x).detach().cpu().numpy()
            
            score = np.append(score, output, axis=0)
    
    C_Index = c_index(score, T_true, E_true, num_times)
    IBS = ibs(score, T_true, E_true, num_times, times)
    
    return C_Index, IBS