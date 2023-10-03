import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, precision_score, recall_score, f1_score

def get_DAG(B):
    A = np.where(B!=0,1,0)
    return A

def count_accuracy_complete(B_true, B_est): 
    """Compute various accuracy metrics for B_est.
    
    INPUT
    =====
    B_true: np.ndarray, [J,L+1,N,N] ground truth graph, {0, 1}
    B_est: np.ndarray, [J,L+1,N,N] estimate, {0, 1}
    
    OUTPUT
    ======
    acc: float, accuracy
    'fdr|(1-precision)': float, 1-precision
    'tpr|recall': float, recall
    f1: float, f1-score
    fpr: float, false positive rate
    nnz: int, number of predicted edges
    true_nnz: int, number of edges in the ground truth
    nnz/true_nnz: int, predicted network size
    shd-missing: int, number of missing edges
    shd-extra': int, number of predicted extra edges (those that are not in the skeleton of the ground truth) 
    shd-reverse: int, number of predicted reverse edges
    shd: int, structural Hamming distance (undirected extra + undirected missing + reverse)
    shd/true_nnz: float, shd divided by the number of edges in the ground truth
    """

    B_true, B_est = get_DAG(B_true),get_DAG(B_est)
    
    J, L, _, _ = B_est.shape

    #check values
    
    if not ((B_true == 0) | (B_true == 1)).all():
        raise ValueError('B_true should take value in {0,1}')

    if not ((B_est == 0) | (B_est == 1)).all():
        raise ValueError('B_est should take value in {0,1}')
    
    #B_est and B_true will have shapes [J,L,N,N]
    
    y_pred = B_est.flatten()
    y_true = B_true.flatten()
    
    nnz = len(np.flatnonzero(B_est))
    true_nnz = len(np.flatnonzero(B_true))
    
    acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='binary', zero_division=1)
    rec = recall_score(y_true, y_pred, average='binary', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=1)
    
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp/(fp+tn)

    #structural Hamming distance
    extra_lower, missing_lower, reverse= np.array([]), np.array([]), np.array([])
    
    for j in range(J):
        for l in range(L):
            pred_jl = np.flatnonzero(B_est[j,l])
            cond_jl = np.flatnonzero(B_true[j,l])

            if l==0:
                #in case of instantaneous interactions
                #we need to take into account reverse edges as well.    
                cond_reversed_jl = np.flatnonzero(B_true[j,l].T)
                # cond_skeleton_jl = np.concatenate([cond_jl, cond_reversed_jl])

                extra_rev_jl = np.setdiff1d(pred_jl, cond_jl, assume_unique=True)
                reverse_jl = np.intersect1d(extra_rev_jl, cond_reversed_jl, assume_unique=True)
                
                #in this case we need undirected edges as well.
                pred_lower_jl = np.flatnonzero(np.tril(B_est[j,l] + B_est[j,l].T, k=-1))
                cond_lower_jl = np.flatnonzero(np.tril(B_true[j,l] + B_true[j,l].T, k=-1))
                extra_lower_jl = np.setdiff1d(pred_lower_jl, cond_lower_jl, assume_unique=True)
                missing_lower_jl = np.setdiff1d(cond_lower_jl, pred_lower_jl, assume_unique=True)
            
            elif l>0:
                #in case of lagged interactions
                #reversed edges are not allowed 
                #due to time ordering.
                #Therefore undirected edges have
                #an implicit direction  
                cond_skeleton_jl = np.copy(cond_jl)

                extra_lower_jl = np.setdiff1d(pred_jl, cond_skeleton_jl, assume_unique=True)
                missing_lower_jl = np.setdiff1d(cond_skeleton_jl, pred_jl, assume_unique=True)
                reverse_jl = np.array([])

            extra_lower = np.append(extra_lower, extra_lower_jl)
            missing_lower = np.append(missing_lower, missing_lower_jl)
            reverse = np.append(reverse, reverse_jl)

    shd = len(extra_lower) + len(missing_lower) + len(reverse)

    return {'accuracy':acc, 'fdr|(1-precision)': 1-prec, 'tpr|recall': rec, 'fpr': fpr, 'f1':f1,
            'nnz': nnz, 'true_nnz':true_nnz,
            'shd-missing':len(missing_lower), 'shd-extra':len(extra_lower), 'shd-reverse':len(reverse), 
            'shd': shd,'shd/true_nnz': shd*1/max(true_nnz,1.)}
