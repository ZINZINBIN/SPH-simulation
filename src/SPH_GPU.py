import numba
import numpy as np
import time
from typing import Dict, Optional, Callable, List
from src.kernels import compute_kernel
from src.NNPS import NNPS
from src.correct import KGC, DFPM
from src.force import compute_pressure, compute_du_viscous, compute_du_pressure, compute_du_gravity
from src.visualize import Monitor
from tqdm.auto import tqdm
from src.utils import seed_everything, to_numpy_array
import multiprocessing as mp
from functools import partial
import gc, ctypes

# define function used for GPU computation

# SPH solver