''' Kernels for SPH
    - Input : r: [m,d] (m : number of particle, d : dimensions), h : distance, kernel_type : type of kernels
    - Output : kernel W(x,x') and Gradient dW(x,x')
        W(x,x') : [m,1]
        dW(x,x') : [m,d]
'''
import numpy as np
from typing import Literal, Union

def compute_kernel(r:Union[np.ndarray, np.array], h : float, kernel_type : Literal['gaussian', 'quartic', 'wendland2','wendland4','wendland6']='gaussian'):
    
    if type(r) == np.array: # type == np.array due to... dimension == 1
        r = r.reshape(-1,1)
        
    dims = r.shape[1]
    r_abs = np.linalg.norm(r, axis = 1)
        
    if kernel_type == 'gaussian':
        C = 1 / np.sqrt(np.pi) / h
        C = C ** dims
        W_Wd = C * np.exp(-r_abs*r_abs/h**2)
        dW_Wd = C*(-2)*np.multiply(r, np.exp(-r_abs*r_abs/h**2).reshape(-1,1))
        
    elif kernel_type == 'quartic':
        pass
    elif kernel_type == 'wendland2':
        pass
    elif kernel_type == 'wendland4':
        pass
    elif kernel_type == 'wendland6':
        pass
    
    return W_Wd, dW_Wd