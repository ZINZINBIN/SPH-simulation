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
        if dims == 1:
            alpha = 5 / 8 / h
        elif dims == 2:
            alpha = 7 / 4 / np.pi / h**2
        else:
            alpha = 21 / 16 / np.pi / h**2
        pass
    
    elif kernel_type == 'wendland2':
        if dims == 1:
            alpha = 5 / 8 / h
            W_Wd = alpha * (1-r_abs/2/h)**3 * (1+3*r_abs/2/h)
            dW_Wd = np.multiply(r, (alpha/2/h*(-12*(1-r_abs/2/h)**2)/2/h).reshape(-1,1))
        elif dims == 2:
            alpha = 7 / 4 / np.pi / h**2
            W_Wd = alpha * (1-r_abs/2/h)**3 * (1+3*r_abs/2/h)
            dW_Wd = np.multiply(r, (alpha/2/h*(-20*(1-r_abs/2/h)**3)/2/h).reshape(-1,1))
        else:
            alpha = 21 / 16 / np.pi / h**2
            W_Wd = alpha * (1-r_abs/2/h)**3 * (1+3*r_abs/2/h)
            dW_Wd = np.multiply(r, (alpha/2/h*(-20*(1-r_abs/2/h)**3)/2/h).reshape(-1,1))
            
    elif kernel_type == 'wendland4':
        if dims == 1:
            alpha = 3 / 4 / h
            W_Wd = alpha * (1-r_abs/2/h)**5 * (1+5*r_abs/2/h+8*(r_abs/2/h)**2)
            dW_Wd = np.multiply(r, (alpha/2/h*(-14*(1-r_abs/2/h)**4 *(1+4*r_abs/2/h))/2/h).reshape(-1,1))
        elif dims == 2:
            alpha = 9 / 4 / np.pi / h**2
            W_Wd = alpha * (1-r_abs/2/h)**6 * (1+6*r_abs/2/h+35/3*(r_abs/2/h)**2)
            dW_Wd = np.multiply(r, (alpha/2/h*(-56/3*(1-r_abs/2/h)**5 *(1+5*r_abs/2/h))/2/h).reshape(-1,1))
        else:
            alpha = 495 / 256/ h**3
            W_Wd = alpha * (1-r_abs/2/h)**6 * (1+6*r_abs/2/h+35/3*(r_abs/2/h)**2)
            dW_Wd = np.multiply(r, (alpha/2/h*(-56/3*(1-r_abs/2/h)**5 *(1+5*r_abs/2/h))/2/h).reshape(-1,1))
    
    return W_Wd, dW_Wd