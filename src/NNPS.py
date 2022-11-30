"""Nearest Neighbor Paritcle Search algorithms
    Input : [r,ri,r_sup]
    Output : [adj_indices]
    Argument:
    - r : np.ndarray for particles positions
    - ri : np.ndarray for particles positions used for search
    - r_sup : maximum number of neighbor particles for search
    
"""

import numpy as np
from typing import Union

def NNPS(r : np.ndarray, ri : Union[np.ndarray, np.array], r_sup : float):
    m,n = r.shape # m : number of particle, n : dimension
    
    r_abs = np.linalg.norm(r - ri, axis = 1)
    adj_indices = np.where(r_abs < r_sup)[0]

    return adj_indices