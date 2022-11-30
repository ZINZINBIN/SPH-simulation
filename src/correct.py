import numpy as np
from typing import Union

# Kernel Gradient Correction
def KGC(idx : int, adj_idx : np.array, r : np.ndarray, W_Wd : np.ndarray, dW_Wd:np.ndarray, m : Union[np.ndarray, np.array], rho : Union[np.ndarray, np.array]):
    # dW_Wd : np.ndarray, [adj_idx, d]
    
    # ignore the case where the adjacency particle does not exist -> Inverse matrix not exist
    if len(adj_idx) >= 1:
        return W_Wd, dW_Wd
    
    dims = r.shape[1]
    dV = m / rho
    L = np.zeros((dims,dims))
    
    for idx_i in range(0,dims):
        for idx_j in range(0,dims):
            L[idx_i, idx_j] = np.sum((r[adj_idx,idx_j] - r[idx,idx_j])*dW_Wd[:,idx_i]*dV[adj_idx])
    
    try:
        L_inv = np.linalg.inv(L)
    except:
        L_inv = np.ones_like(L)
    
    W_Wd_cor = W_Wd
    dW_Wd_cor = np.matmul(L_inv, dW_Wd.T).T
    
    return W_Wd_cor, dW_Wd_cor

# Corrective smoothed particle method : CSPM
def CSPM():
    pass

# Finite Point Method
def FPM(idx : int, adj_idx : np.array, r : np.ndarray, W_Wd : np.ndarray, dW_Wd:np.ndarray, m : Union[np.ndarray, np.array], rho : Union[np.ndarray, np.array]):
    
    dims = r.shape[1]
    dV = m / rho
    L = np.zeros((dims+1,dims+1))
            
    for idx_i in range(0,dims):
        for idx_j in range(0,dims):
            L[idx_i+1, idx_j+1] = np.sum((r[adj_idx,idx_j] - r[idx_i,idx_j])*dW_Wd[:,idx_i]*dV[adj_idx])
            
    for idx_j in range(0,dims):
        L[0,idx_j+1] = np.sum((r[adj_idx,idx_j] - r[idx_i,idx_j])*W_Wd*dV[adj_idx])
        
    for idx_i in range(0,dims):
        L[idx_i+1,0] = np.sum(dW_Wd[:,idx_j]*dV[adj_idx])
        
    L[0,0] = np.sum(W_Wd * dV[adj_idx])
    
    L_inv = np.linalg.inv(L)
    
    cor = np.matmul(L_inv, np.concatenate(W_Wd, dW_Wd, axis = 1).T).T
    W_Wd_cor = cor[:,0]
    dW_Wd_cor = cor[:,1:]
    return W_Wd_cor, dW_Wd_cor
    
# Decomposed Finite Point Method
def DFPM(idx : int, adj_idx : np.array, r : np.ndarray, W_Wd : np.ndarray, dW_Wd:np.ndarray, m : Union[np.ndarray, np.array], rho : Union[np.ndarray, np.array]):
    
    dims = r.shape[1]
    dV = m / rho
    
    W_filter = np.sum(W_Wd * dV[adj_idx], axis = 0)
    dW_Wd_filter = np.sum((r[adj_idx,:] - r[idx,:]).reshape(-1,dims) * dW_Wd * dV[adj_idx].reshape(-1,1), axis = 0)
    
    W_Wd_cor = W_Wd / W_filter
    dW_Wd_cor = dW_Wd / dW_Wd_filter
    
    return W_Wd_cor, dW_Wd_cor
    
# Kernel Gradient Free SPH 
def KGFSPH(idx : int, adj_idx : np.array, r : np.ndarray, W_Wd : np.ndarray, dW_Wd:np.ndarray, m : Union[np.ndarray, np.array], rho : Union[np.ndarray, np.array]):
    
    dims = r.shape[1]
    dV = m / rho
    
    W_filter = np.sum(W_Wd * dV[adj_idx], axis = 0)
    dW_Wd_filter = np.sum((r[adj_idx,:] - r[idx,:]).reshape(-1,dims) * dW_Wd * dV[adj_idx].reshape(-1,1), axis = 0)
    
    W_Wd_cor = W_Wd / W_filter
    dW_Wd_cor = dW_Wd / dW_Wd_filter
    
    return W_Wd_cor, dW_Wd_cor