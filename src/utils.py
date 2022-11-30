import numpy as np
import random, os
from numba import cuda

def seed_everything(seed_num : int = 42, use_nn : bool = False)->None:
    
    print("# initialize random seed number")
    random.seed(seed_num)
    np.random.seed(seed_num)
    
     # os environment seed num fixed
    try:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
    except:
        pass
    
def to_numpy_array(shared_array, shape):
    arr = np.ctypeslib.as_array(shared_array)
    return arr.reshape(shape)

# GPU check
def check_gpu():
    print("gpu check : ", cuda.gpus)

def select_gpu(device_id : int):
    cuda.select_device(device_id)