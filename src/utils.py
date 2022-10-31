import numpy as np
import random, os

def seed_everything(seed_num : int = 42, use_nn : bool = False)->None:
    
    print("# initialize random seed number")
    random.seed(seed_num)
    np.random.seed(seed_num)
    
     # os environment seed num fixed
    try:
        os.environ["PYTHONHASHSEED"] = str(seed_num)
    except:
        pass