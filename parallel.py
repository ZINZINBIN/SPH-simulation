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
import argparse  

parser = argparse.ArgumentParser(description="2D SPH simulation as my own project with multi-processing")
parser.add_argument("--height", type = float, default = 2.0)
parser.add_argument("--width", type = float, default = 15.0)
parser.add_argument("--num_particle", type = int, default = 1024)
parser.add_argument("--end_time", type = float, default = 5.0)
parser.add_argument("--radius", type = float, default = 0.0125)
parser.add_argument("--dt", type = float, default = 1e-2)
parser.add_argument("--mu", type = float, default = 1.0 * 10 **(-1))
parser.add_argument("--density", type = float, default = 1000.0)
parser.add_argument("--sound_speed", type = float, default = 32)
parser.add_argument("--g", type = float, default = 9.8)
parser.add_argument("--kn", type = int, default = 2)
parser.add_argument("--tag", type = str, default = "simulation-parallel")
parser.add_argument("--kernel_type", type = str, default = "gaussian")
parser.add_argument("--cor_type", type = str, default = "KGC")
parser.add_argument("--use_bp", type = bool, default = False)
parser.add_argument("--plot_freq", type = int, default = 50)
parser.add_argument("--plot_boundary_particle", type = bool, default = False)
parser.add_argument("--verbose", type = bool, default = True)

# For 2D SPH solver
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
        kernel_type : str = 'gaussian',
        cor_type : str = 'DFPM',
        use_bp : bool = False,
        plot_freq : int = 10,
        plot_boundary_particle : bool = False,
        boundary_info : Optional[Dict] = None,
        save_dir : str = "./result/simulation.gif",
        verbose : bool = False
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
        P_shared = mp.Array(ctypes.c_double, self.Nt, lock=False)
        rho_shared = mp.Array(ctypes.c_double, self.Nt, lock=False)
        u_shared = mp.Array(ctypes.c_double, self.Nt * self.boundary_info['dims'], lock=False)
        
        self.P = to_numpy_array(P_shared, (self.Nt,))
        self.rho = to_numpy_array(rho_shared, (self.Nt,))
        
        for idx in range(0,self.Nt):
            self.rho[idx] = rho
        
        self.u = to_numpy_array(u_shared, (self.Nt, self.boundary_info['dims']))
        
        self.rho0 = np.ones((self.Nt,)) * rho
        self.mu = np.ones((self.Nt,)) * mu
        self.m = np.ones((self.Nt,)) * rho * radius ** 2 * np.pi
        
        self.C = C
        self.g = g
        
        steps = round((tf-ti)/dt)
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
        
        # multi-processing
        self.n_jobs = mp.cpu_count()
        print("multi-processing : {}-cpu will be used".format(self.n_jobs))
        
    def compute_per_process(self, idx : int, exec : Callable):
        adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
        W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
        exec(idx, adj_idx, W_Wd, dW_Wd)       
    
    def compute_multi_process(self, exec : Callable):
          
        particles = [idx for idx in range(0,self.Nt)]
        
        compute_per_procs = partial(
            self.compute_per_process,
            exec = exec
        )
        
        # use Process class
        processes = []
        for p in particles:
            process = mp.Process(target = compute_per_procs, args = (p,))
            processes.append(process)
        
        [proc.start() for proc in processes]
        [proc.join() for proc in processes]
        

    def animate(self):
        self.monitor.animate(self.update)
        
    def initialize_boundary(self):
        
        if self.use_bp:
            
            h = self.boundary_info['height']
            w = self.boundary_info['width']
            dims = self.boundary_info['dims']
            pad = self.boundary_info['pad']
            
            self.Nb = int(h/self.vol)*2 + int(w/self.vol)
            self.Nb *= pad
            self.Nt = self.Np + self.Nb
            
            r_shared = mp.Array(ctypes.c_double, self.Nt*dims, lock=False)
            ptype_shared = mp.Array(ctypes.c_double, self.Nt, lock=False)
            
            self.r = to_numpy_array(r_shared, (self.Nt, dims))
            self.ptype = to_numpy_array(ptype_shared, (self.Nt,))
            
            for i in range(self.Nt):
                self.ptype[i] = 1
            
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
            
            r_shared = mp.Array(ctypes.c_double, self.Nt*dims, lock=False)
            ptype_shared = mp.Array(ctypes.c_double, self.Nt, lock=False)
            
            self.r = to_numpy_array(r_shared, (self.Nt, dims))
            self.ptype = to_numpy_array(ptype_shared, (self.Nt,))
            
            for i in range(self.Nt):
                self.ptype[i] = 1
    
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
            self.u[idx,0] *= (-1)
            
        elif self.r[idx,0] < 0:
            self.r[idx,0] = 0
            self.u[idx,0] *= (-1)
            
        if self.r[idx,1] < 0:
            self.r[idx,1] = 0
            self.u[idx,1] *= (-1)
            
    
if __name__ == "__main__":

    args = vars(parser.parse_args())

    Boundary_Info = {
        "height" : args['height'],
        "width": args["width"],
        "dims" : 2,
        "pad":8,
    }
    
    solver = SPHsolver(
        Np = args['num_particle'],
        ti = 0,
        tf = args['end_time'],
        dt = args['dt'],
        mu = args['mu'],
        rho = args['density'],
        C = args['sound_speed'],
        g = args['g'],
        radius = args['radius'],
        kn = args['kn'],    
        kernel_type=args['kernel_type'],
        cor_type = args['cor_type'],
        use_bp = args['use_bp'],
        plot_freq = args['plot_freq'],
        plot_boundary_particle=args['plot_boundary_particle'],
        boundary_info = Boundary_Info,
        save_dir = "./result/{}.gif".format(args['tag']),
        verbose = args['verbose']
    )
    
    solver.animate()