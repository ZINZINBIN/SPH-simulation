import numpy as np
import ctypes, time, math
import multiprocessing as mp
import matplotlib.pyplot as plt
from numba import jit, prange, cuda, float32

def func_a(idx : int):
    a[idx] += idx
    print("a[{}]:{}".format(idx,a[idx]))

def func_b(idx : int):
    b[idx] += idx
    print("b[{}]:{}".format(idx,b[idx]))
    
    b[idx] += c[idx]
    print("b[{}] + c[{}]:{}".format(idx,idx,b[idx]))
    
# parallel performance comparison
@jit(nopython = True)
def func1(data, box_size):
    res = np.empty_like(data)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            res[y,x] = 0
            for yy in range(0, box_size):
                for xx in range(0,box_size):
                    nx = x + xx - int(box_size / 2)
                    ny = y + yy - int(box_size / 2)
                    
                    if nx < 0 : 
                        nx = 0
                    elif nx >= res.shape[1]:
                        nx = res.shape[1] - 1
                    
                    if ny < 0:
                        ny = 0
                    elif ny >= res.shape[0]:
                        ny = res.shape[0] - 1
                    
                    res[y,x] += data[ny,nx] / (box_size * box_size)
    return res

@jit(nopython = True, parallel = True)
def func2(data, box_size):
    res = np.empty_like(data)
    for y in range(res.shape[0]):
        for x in range(res.shape[1]):
            res[y,x] = 0
            for yy in range(0, box_size):
                for xx in range(0,box_size):
                    nx = x + xx - int(box_size / 2)
                    ny = y + yy - int(box_size / 2)
                    
                    if nx < 0 : 
                        nx = 0
                    elif nx >= res.shape[1]:
                        nx = res.shape[1] - 1
                    
                    if ny < 0:
                        ny = 0
                    elif ny >= res.shape[0]:
                        ny = res.shape[0] - 1
                    
                    res[y,x] += data[ny,nx] / (box_size * box_size)
    return res

@cuda.jit
def func3(data, box_size, res):
    y,x = cuda.grid(2)
    
    if y < res.shape[0] and x < res.shape[1]:
        res[y,x] = 0
        for yy in range(0, box_size):
            for xx in range(0,box_size):
                nx = x + xx - int(box_size / 2)
                ny = y + yy - int(box_size / 2)
                
                if nx < 0 : 
                    nx = 0
                elif nx >= res.shape[1]:
                    nx = res.shape[1] - 1
                
                if ny < 0:
                    ny = 0
                elif ny >= res.shape[0]:
                    ny = res.shape[0] - 1
                
                res[y,x] += data[ny,nx] / (box_size * box_size)   
        

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
    
    # using numba
    # jit test
    a = np.zeros((1000,1000))
    def no_jit_test():
        rho = np.zeros_like(a)
        print("rho : ", rho.shape)
        
    start_time = time.time()
    no_jit_test()
    end_time = time.time()
    
    print("no jit, dt : ", end_time - start_time)
    
    @jit
    def jit_test():
        rho = np.zeros_like(a)
        print("rho : ", rho.shape)
    
    start_time = time.time()
    jit_test()
    end_time = time.time()
    
    print("jit, dt : ", end_time - start_time)
    
    # GPU check
    def check_gpu():
        print("gpu check : ", cuda.gpus)

    def select_gpu(device_id : int):
        cuda.select_device(device_id)
    
    check_gpu()    
    select_gpu(0)
    
    print("# test for numba parallel computing #")
    
    @cuda.jit
    def matmul(A,B,C):
        i,j = cuda.grid(2)
        if i < C.shape[0] and j < C.shape[1]:
            tmp = 0
            for k in range(A.shape[1]):
                tmp += A[i,k] * B[k,j]
            C[i,j] = tmp
            
        cuda.syncthreads()
    
    @cuda.jit
    def fast_matmul(A,B,C):
        sA = cuda.shared.array(shape = (TPB, TPB), dtype = float32)
        sB = cuda.shared.array(shape = (TPB, TPB), dtype = float32)
        
        # sA = cuda.device_array(shape = (TPB,TPB), dtype = float32)
        # sB = cuda.device_array(shape = (TPB,TPB), dtype = float32)
        
        x,y = cuda.grid(2)
        tx = cuda.threadIdx.x
        ty = cuda.threadIdx.y
        bpg = cuda.gridDim.x
        
        if x >= C.shape[0] and y >= C.shape[1]:
            return

        tmp = 0
        
        for i in range(bpg):
            
            # preload data into shared memory
            sA[tx,ty] = A[x, ty + i * TPB]
            sB[tx,ty] = B[tx+i*TPB, y]
            
            # wait until all threads finish preloading
            cuda.syncthreads()
            
            for j in range(TPB):
                tmp += sA[tx,j] * sB[j,ty]
            
            cuda.syncthreads()
            
        C[x,y] = tmp
        
        return
    
    SIZE = 1000
    BOX_SIZE = 20
    TPB = 32
    
    # sizes = []
    # results = []
    # for size in range(int(SIZE / 10), int(SIZE + SIZE / 10), int(SIZE / 10)):
    #     print(size)

    #     sizes.append(size)
    #     x = np.arange(size * size, dtype=np.float32).reshape((size, size))

    #     s = time.time()
    #     a = func1(x, BOX_SIZE)
    #     t1 = time.time() - s

    #     s = time.time()
    #     xx = np.empty_like(x)

    #     threadsperblock = (TPB, TPB)
    #     blockspergrid_x = int(math.ceil(x.shape[0] / threadsperblock[1]))
    #     blockspergrid_y = int(math.ceil(x.shape[1] / threadsperblock[0]))
    #     blockspergrid = (blockspergrid_x, blockspergrid_y)

    #     x_dary = cuda.to_device(x)
    #     xx_dary = cuda.device_array(xx.shape, xx.dtype)
    #     func3[blockspergrid, threadsperblock](x_dary, BOX_SIZE, xx_dary)
    #     xx_dary.copy_to_host(xx)

    #     t3 = time.time() - s

    #     assert np.all(a == xx)

    #     results.append((t1, t3))

    # plt.plot(sizes, [x[0] for x in results], 'rs-', label='single')
    # plt.plot(sizes, [x[1] for x in results], 's-', label='cuda')
    # plt.legend()
    # plt.savefig('./result/parallel-test.png')
    
    TPB = 64
    
    A = np.random.randn(10000,10000)
    B = np.random.randn(10000,10000)
    C = np.zeros_like(A)
    
    blockdim = (20,20)
    griddim = (5000,5000)
    
    start_time = time.time()
    C = np.matmul(A,B)
    end_time = time.time()
    
    print("numpy : matrix multiplication : {:.5f}".format(end_time - start_time))
    
    A_cuda = cuda.to_device(A)
    B_cuda = cuda.to_device(B)
    C_cuda = cuda.device_array_like(C)
    
    start_time = time.time()
    matmul[griddim, blockdim](A_cuda, B_cuda,C_cuda)
    end_time = time.time()
    
    print("CUDA : matrix multiplication - algorithm 1: {:.5f}".format(end_time - start_time))
    
    start_time = time.time()
    fast_matmul[griddim, blockdim](A_cuda, B_cuda, C_cuda)
    end_time = time.time()
    
    print("CUDA : matrix multiplication - algorithm 2: {:.5f}".format(end_time - start_time))