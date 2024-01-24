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