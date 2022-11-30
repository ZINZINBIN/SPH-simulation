from numba import cuda
import numpy as np
import time
from typing import Dict, Optional, Callable, List, Union
from src.kernels import compute_kernel
from src.force import compute_pressure, compute_du_viscous, compute_du_pressure, compute_du_gravity
from src.visualize import Monitor
from tqdm.auto import tqdm
from src.NNPS import NNPS
from src.utils import seed_everything, to_numpy_array, check_gpu, select_gpu
from functools import partial
import gc

# define function used for GPU computation
@cuda.jit
def NNPS_gpu(r : np.ndarray, ri : Union[np.ndarray, np.array], r_sup : float):
    m,n = r.shape # m : number of particle, n : dimension
    r_abs = np.linalg.norm(r - ri, axis = 1)
    adj_indices = np.where(r_abs < r_sup)[0]
    return adj_indices

# kernel sum for mass computation : use mass conservation laws
@cuda.jit
def update_mass(m, rho, r, u, W_Wd, dW_Wd, idx, adj_idx, dt):
    drho_dt_tmp = rho[idx] * np.sum(m[adj_idx] / rho[adj_idx] * np.sum(dW_Wd * (u[idx] - u[adj_idx]), axis = 1), axis = 0)
    rho[idx] += dt * drho_dt_tmp
    
# kernel sum for momentum computation
@cuda.jit     
def update_momentum(m, rho, r, u, W_Wd, dW_Wd, idx, adj_idx, dt, ptype, C, mu, g):
    du_dt = compute_du_pressure(rho, m, C, idx, adj_idx, dW_Wd) + compute_du_viscous(rho, mu, r, m, u, idx, adj_idx, dW_Wd) + compute_du_gravity(u, idx, adj_idx, g)

    u[idx,:] += dt * du_dt * (ptype[idx] == 1)
    r[idx,:] += dt * u[idx,:] * (ptype[idx] == 1)

# kernel for shepard filter : not used in this case
@cuda.jit 
def shepard_filter(m, rho, W_Wd, dW_Wd, idx, adj_idx):
    flt_s = np.sum(m[adj_idx] / rho[adj_idx] * W_Wd, axis = 0)
    W_Wd /= flt_s
    return W_Wd

# reflection condition using elastic collision
@cuda.jit
def correct_reflection(r, u, idx, adj_idx, gamma, width, height):
    w = width

    if r[idx,0] > w:
        r[idx,0] = w
        u[idx,0] *= (-1) * gamma
        
    elif r[idx,0] < 0:
        r[idx,0] = 0
        u[idx,0] *= (-1) * gamma 
        
    if r[idx,1] < 0:
        r[idx,1] = 0
        u[idx,1] *= (-1) * gamma

@cuda.jit
def cuda_kernel(exec : Callable):
    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y
    pass


# For 2D SPH solver with CUDA computing
class SPHsolver:
    def __init__(
        self,  
        Np : int, 
        ti : float = 0,
        tf : float = 4.0,
        dt : float = 1e-3,
        mu : float = 0.1,
        rho : float = 1.0,
        C : float = 100,
        g : float = 9.8,
        radius : float = 0.05,
        kn : int = 2,
        gamma : float = 0.1,
        kernel_type : str = 'gaussian',
        cor_type : str = 'DFPM',
        use_bp : bool = False,
        plot_freq : int = 10,
        plot_boundary_particle : bool = False,
        boundary_info : Optional[Dict] = None,
        save_dir : str = "./result/simulation-gpu.gif",
        verbose : bool = False,
        device : int = 0
        ):
        
        seed_everything(42)
        
        # SPH parameter
        self.Np = Np # number of particle (except boundary)
        self.ti = ti # start time
        self.tf = tf # end time
        self.dt = dt # time difference
        self.radius = radius # particle radius
        self.vol = radius * 2 # grid size
        self.h = 2 * radius * 1.5 # smoothed paramter
        self.kn = kn # number of particle for considering
        self.r_sup = self.h * self.kn # support radius
        self.kernel_type = kernel_type
        self.cor_type = cor_type
        
        # boundary info
        self.use_bp = use_bp
        self.boundary_info = boundary_info
        
        # initialize boundary with 2D shape
        self.initialize_boundary()
        
        # initialize particles
        self.initialize_particles()
        
        # physical parameter
        self.P = np.zeros((self.Nt,))
        self.rho = np.ones((self.Nt,)) * rho
        self.rho0 = np.ones((self.Nt,)) * rho
        self.mu = np.ones((self.Nt,)) * mu
        self.m = np.ones((self.Nt,)) * rho * radius ** 2 * np.pi
        self.u = np.zeros_like(self.r)
        self.du_dt = np.zeros_like(self.r)
        self.C = C
        self.g = g
        self.gamma = gamma
        
        steps = round((tf-ti)/dt) + 1
        self.ts = np.linspace(ti, tf, steps)
        
        # video configuration
        self.plot_freq = plot_freq
        self.plot_boundary_particle = plot_boundary_particle
        self.monitor = Monitor(
            n_particle=Np,
            plot_freq = plot_freq, 
            boundary_info = boundary_info, 
            t_stats = self.ts, 
            figsize = (10,6), 
            blit = True, 
            save_dir=save_dir
        )
        
        self.verbose = verbose
        
        # GPU - computing with CUDA
        self.device = device
        
        # check the state of GPU device
        check_gpu()
        
        # GPU allocation
        select_gpu(device)
    
    def allocate_particle_indice(self):
        # particle index matching with threadIdx
        pass
    
    @cuda.jit
    def compute_per_process(self, r : np.ndarray, idx : int, r_sup : float, kernel_type : str, exec : Callable):
        # call particle index from cuda.threadIdx
        
        
        # For each particle, compute NNPS algorithms and kernel
        adj_idx = NNPS(r,r[idx],r_sup)
        W_Wd, dW_Wd = compute_kernel((-1) * (r[adj_idx] - r[idx]), r_sup, kernel_type)
        
        # Finally, execute the function exec for each particle
        exec(idx, adj_idx, W_Wd, dW_Wd)       
    
    def compute_multi_process(self, exec : Callable):
          
        particles = [idx for idx in range(0,self.Nt)]
        
        # cpu to device
        r_d = cuda.to_device(self.r)
        u_d = cuda.to_device(self.u)
        
        compute_per_procs = partial(
            self.compute_per_process,
            exec = exec
        )

    def animate(self): # single core computation 
        self.monitor.animate(self.update)
        
    def initialize_boundary(self): # initialization : single core computation
        
        if self.use_bp:
            h = self.boundary_info['height']
            w = self.boundary_info['width']
            dims = self.boundary_info['dims']
            pad = self.boundary_info['pad']
            
            self.Nb = int(h/self.vol)*2 + int(w/self.vol)
            self.Nb *= pad
            self.Nt = self.Np + self.Nb
            
            self.r = np.zeros((self.Nt, dims))
            self.ptype = np.ones((self.Nt,))
            
            # with padding
            for idx_pad in range(0, int(pad)):
                for idx in range(0,int(h/self.vol)):
                    self.r[idx + int(h/self.vol) * idx_pad,0] = -self.radius * (2 * idx_pad + 1)
                    self.r[idx + int(h/self.vol) * idx_pad,1] = self.radius * (2 * idx + 1)
                    self.ptype[idx + int(h/self.vol) * idx_pad] = 0
                    
            for idx_pad in range(0, int(pad)):
                for idx in range(0, int(w/self.vol)):
                    self.r[idx + int(h/self.vol) * pad + int(w/self.vol) * idx_pad,0] = self.radius * (2 * idx + 1)
                    self.r[idx + int(h/self.vol) * pad + int(w/self.vol) * idx_pad,1] = -self.radius * (2 * idx_pad + 1)
                    self.ptype[idx + int(h/self.vol) * pad + int(w/self.vol) * idx_pad] = 0
            
            for idx_pad in range(0, int(pad)):
                for idx in range(0,int(h/self.vol)):
                    self.r[idx + int(h/self.vol) * pad + int(w/self.vol) * pad + int(h/self.vol) * idx_pad,1] = self.radius * (2 * idx + 1)
                    self.r[idx + int(h/self.vol) * pad + int(w/self.vol) * pad + int(h/self.vol) * idx_pad,0] = w + self.radius *(2 * idx_pad + 1)
                    self.ptype[idx + int(h/self.vol) * pad+ int(w/self.vol) * pad + int(h/self.vol) * idx_pad] = 0
                    
        else:
            self.Nt = self.Np
            dims = self.boundary_info['dims']
            
            self.r = np.zeros((self.Nt, dims))
            self.ptype = np.ones((self.Nt,))
    
    def initialize_particles(self):
        
        h = self.radius
        w = self.radius
        ratio = 0.8
        
        for idx in range(self.Nt):
            
            if self.ptype[idx] == 1:
                self.r[idx,:] = np.array([w,h])
                h += self.radius * 2
                
            if h >= self.boundary_info['height'] * ratio:
                h = self.radius
                w += self.radius * 2
                
    def update(self, t):
        
        if self.verbose:
            start_time = time.time()
        
        # mass conservation
        self.compute_multi_process(self.update_mass)
         
        # update pressure
        self.P = compute_pressure(self.rho, self.rho0, self.C)
            
        # momentum condition
        self.compute_multi_process(self.update_momentum)
        
        # correct reflection
        self.compute_multi_process(self.correct_reflection)
     
        # animation
        if self.plot_boundary_particle:
            self.monitor.update(self.r[self.ptype==1], self.r[self.ptype==0])
        else:
            self.monitor.update(self.r[self.ptype==1], None)
            
        if self.verbose:
            end_time = time.time()
            print("# t = {:.3f}, run time : {:.3f}, P : {:.3f}, rho : {:.3f}, u : {:.3f}".format(t, end_time - start_time, np.average(self.P), np.average(self.rho), np.average(self.u)))
            
        # empty cache
        gc.collect()
        
        return self.monitor.points,
    
    def update_mass(self, idx, adj_idx, W_Wd, dW_Wd):
        drho_dt_tmp = self.rho[idx] * np.sum(self.m[adj_idx] / self.rho[adj_idx] * np.sum(dW_Wd * (self.u[idx] - self.u[adj_idx]), axis = 1), axis = 0)
        self.rho[idx] += self.dt * drho_dt_tmp
    
    def update_momentum(self,idx, adj_idx, W_Wd, dW_Wd):
        du_dt = compute_du_pressure(self.rho, self.m, self.C, idx, adj_idx, dW_Wd) + compute_du_viscous(self.rho, self.mu, self.r, self.m, self.u, idx, adj_idx, dW_Wd) + compute_du_gravity(self.u, idx, adj_idx, self.g)
 
        self.u[idx,:] += self.dt * du_dt * (self.ptype[idx] == 1)
        self.r[idx,:] += self.dt * self.u[idx,:] * (self.ptype[idx] == 1)
    
    def shepard_filter(self, adj_idx : np.array, W_Wd : np.ndarray):
        flt_s = np.sum(self.m[adj_idx] / self.rho[adj_idx] * W_Wd, axis = 0)
        W_Wd /= flt_s
        return W_Wd
    
    def correct_reflection(self, idx, adj_idx, W_Wd, dW_Wd):
        
        w = self.boundary_info['width']

        if self.r[idx,0] > w:
            self.r[idx,0] = w
            self.u[idx,0] *= (-1) * self.gamma
            
        elif self.r[idx,0] < 0:
            self.r[idx,0] = 0
            self.u[idx,0] *= (-1) * self.gamma
            
        if self.r[idx,1] < 0:
            self.r[idx,1] = 0
            self.u[idx,1] *= (-1) * self.gamma