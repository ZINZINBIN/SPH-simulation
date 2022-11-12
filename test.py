import numpy as np
import ctypes
import multiprocessing as mp

def func_a(idx : int):
    a[idx] += idx
    print("a[{}]:{}".format(idx,a[idx]))

def func_b(idx : int):
    b[idx] += idx
    print("b[{}]:{}".format(idx,b[idx]))
    
    b[idx] += c[idx]
    print("b[{}] + c[{}]:{}".format(idx,idx,b[idx]))
        

if __name__ == "__main__":
    a = np.array([0,0,0,0,0,0,0,0])
    b_mp = mp.Array(ctypes.c_double, 8, lock=False)
    c = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
    b = np.ctypeslib.as_array(b_mp).reshape(8,1)
    pool = mp.Pool(processes=4)
    
    print("a : ", a)
    pool.map(func_a, [0,1,2,3,4,5,6,7])
    pool.close()
    pool.join()
    
    print("a : ", a)
    print("b : ", b)
    
    pool = mp.Pool(processes=4)
    pool.map(func_b, [0,1,2,3,4,5,6,7])
    pool.close()
    pool.join()
    
    print("b : ", b)
    print("b : ", b.reshape(4,2))
    
    class test:
        def __init__(self):
            arr = mp.Array(ctypes.c_double, 8, lock=False)
            self.arr = np.ctypeslib.as_array(arr)
        
        def compute(self, idx:int):
            self.arr[idx] += idx
            
        def compute_mp(self):
            pool = mp.Pool(processes=4)
            
            processes = []
            def compute(idx:int):
                self.compute(idx)
                
            for idx in range(0,8):
                process = mp.Process(target = compute, args = (idx,))
                processes.append(process)
            
            [x.start() for x in processes]
            [x.join() for x in processes]
            
        
    class_test = test()
    print("before test : ", class_test.arr)
    
    class_test.compute_mp()
    
    print("after test : ", class_test.arr)
    
    class_test.compute_mp()
    
    print("after test : ", class_test.arr)