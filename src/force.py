import numpy as np
from typing import Union

def compute_pressure(rho : np.array, rho0 : np.array, C : float):
    p = (rho - rho0) * C**2
    return p

def compute_du_pressure(rho : np.array, m : np.array, C : float, idx : Union[int, np.array], adj_idx : np.array, dW_Wd : np.ndarray):
    dp = m[adj_idx]*C*C*(rho[idx] + rho[adj_idx])/rho[idx]/rho[adj_idx]
    du_dt = np.sum(dp.reshape(-1,1)*dW_Wd, axis = 0) * (-1)
    return du_dt
    
def compute_du_viscous(rho : np.array, mu : np.array, r : np.ndarray, m : np.array, u : np.ndarray, idx : Union[int, np.array], adj_idx : np.array, dW_Wd : np.ndarray):
    eps = 1e-6
    r_rel = r[idx,:] - r[adj_idx,:]
    r_abs = np.linalg.norm(r[adj_idx,:] - r[idx,:], axis = 1)
    rW = np.sum(r_rel * dW_Wd, axis = 1)
    partial = 2 * mu[idx] * m[adj_idx] / rho[adj_idx] / rho[idx] * rW / (r_abs * r_abs + eps)
    du_dt = np.sum(partial.reshape(-1,1) * (u[adj_idx,:] - u[idx,:]), axis = 0) * (-1)
    return du_dt
    
def compute_du_gravity(u : np.ndarray, idx : int, adj_idx : np.array, g : float):
    du_dt = np.zeros_like(u[idx])
    du_dt[-1] = g * (-1)
    return du_dt