import numpy as np
import time
from typing import Dict, Optional
from src.kernels import compute_kernel
from src.NNPS import NNPS
from src.correct import KGC, DFPM
from src.force import compute_pressure, compute_du_viscous, compute_du_pressure, compute_du_gravity
from src.visualize import Monitor
from tqdm.auto import tqdm
from src.utils import seed_everything

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
        gamma : float = 0.1,
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
        self.P = np.zeros((self.Nt,), dtype = float)
        self.rho = np.ones((self.Nt,), dtype = float) * rho
        self.rho0 = np.ones((self.Nt,), dtype = float) * rho
        self.mu = np.ones((self.Nt,), dtype = float) * mu
        self.m = np.ones((self.Nt,), dtype = float) * rho * radius ** 2 * np.pi
        self.u = np.zeros_like(self.r, dtype = float)
        self.du_dt = np.zeros_like(self.r, dtype = float)
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
            
        
        self.particle_indices = np.array([i for i in range(self.Nt)], dtype = np.int32)
    
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
        self.update_mass()
        
        # boundary condition
        # self.update_boundary() 
        
        # momentum condition
        self.update_momentum()
        
        # correct reflection
        self.correct_reflection()
            
        # animation
        if self.plot_boundary_particle:
            self.monitor.update(self.r[self.ptype==1], self.r[self.ptype==0])
        else:
            self.monitor.update(self.r[self.ptype==1], None)
            
        if self.verbose:
            end_time = time.time()
            print("# t={:.3f}, run time:{:.3f}, P:{:.3f}, rho:{:.3f}, u:{:.3f}".format(t,end_time - start_time, np.average(self.P), np.average(self.rho), np.average(self.u)))
            
        return self.monitor.points,
    
    def solve(self):
        
        count = 0
        
        for t in tqdm(self.ts):
            # mass conservation
            self.update_mass()
                
            # boundary condition
            # self.update_boundary() 
            
            # momentum condition
            self.update_momentum()
            
            # correct reflection
            self.correct_reflection()
            
            if count % self.plot_freq == 0:
                print("# t={:.3f}, P:{:.3f}, rho:{:.3f}, u:{:.3f}".format(t, np.average(self.P), np.average(self.rho), np.average(self.u)))
                
            count += 1
            
    def update_mass(self): # using continuity equation
        
        rho = np.zeros_like(self.rho, dtype = float)
        drho_dt = np.zeros_like(self.rho, dtype = float)
        P = np.zeros_like(self.P, dtype = float)
        
        for idx in self.particle_indices:
            adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
            W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
            W_Wd, dW_Wd = KGC(idx, adj_idx, self.r, W_Wd, dW_Wd, self.m, self.rho)
            drho_dt[idx] = self.rho[idx] * np.sum(self.m[adj_idx] / self.rho[adj_idx] * np.sum(dW_Wd * (self.u[idx] - self.u[adj_idx]), axis = 1), axis = 0)
            
        # rho = self.rho + self.dt * drho_dt * (self.ptype == 1)
        rho = self.rho + self.dt * drho_dt
        P = compute_pressure(rho, self.rho0, self.C)
        
        self.rho = rho
        self.P = P
    
    def update_boundary(self):
        
        # No-Penetration condition
        for idx in self.particle_indices:
            if self.ptype[idx] <= 0:
                adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
                W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
                flt = np.sum(self.m[adj_idx] / self.rho[adj_idx] * W_Wd * (self.ptype[adj_idx] == 1), axis = 0)
                
                if flt >= 0.005:
                    m_rho = self.m[adj_idx] / self.rho[adj_idx]
                    m_rho = m_rho.reshape(-1,1)
                    is_particle = (self.ptype[adj_idx] == 1).reshape(-1,1)
                    self.u[idx, :] = (-1) * np.sum(m_rho * self.u[adj_idx, :] * W_Wd.reshape(-1,1) * is_particle, axis = 0) / flt
              
        # Neumann Boundary condition     
        for idx in self.particle_indices:
            if self.ptype[idx] <= 0:
                adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
                W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
                flt = np.sum(self.m[adj_idx] / self.rho[adj_idx] * W_Wd * (self.ptype[adj_idx] == 1), axis = 0)
                
                if flt >= 0.005:
                    m_rho = self.m[adj_idx] / self.rho[adj_idx]
                    m_rho = m_rho.reshape(-1,1)
                    is_particle = (self.ptype[adj_idx] == 1).reshape(-1,1)
                    
                    # self.P[idx] = np.sum(self.P[adj_idx] * self.m[adj_idx] / self.rho[adj_idx] * W_Wd.reshape(-1,1) * is_particle) / flt + np.sum(self.rho[adj_idx] * (self.r[adj_idx, 1] - self.r[idx, 1]) * self.g)
                    self.P[idx] = np.sum(self.P[adj_idx] * self.m[adj_idx] / self.rho[adj_idx] * W_Wd.reshape(-1,1) * is_particle) + np.sum(self.rho[adj_idx] * (self.r[adj_idx, 1] - self.r[idx, 1]) * self.g)
                    
                    self.rho[idx] = self.P[idx] / self.C**2 + self.rho0[idx]
        
    def update_momentum(self):
        
        for idx in self.particle_indices:
            adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
            W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
            W_Wd, dW_Wd = KGC(idx, adj_idx, self.r, W_Wd, dW_Wd, self.m, self.rho)

            du_dt = compute_du_pressure(self.rho, self.m, self.C, idx, adj_idx, dW_Wd) + compute_du_viscous(self.rho, self.mu, self.r, self.m, self.u, idx, adj_idx, dW_Wd) + compute_du_gravity(self.u, idx, adj_idx, self.g)
 
            self.u[idx,:] = self.u[idx,:] + self.dt * du_dt * (self.ptype[idx] == 1).reshape(-1,1)
            self.r[idx,:] = self.r[idx,:] + self.dt * self.u[idx,:] * (self.ptype[idx] == 1).reshape(-1,1)
     
    def shepard_filter(self, adj_idx : np.array, W_Wd : np.ndarray):
        flt_s = np.sum(self.m[adj_idx] / self.rho[adj_idx] * W_Wd, axis = 0)
        W_Wd /= flt_s
        return W_Wd
    
    def correct_reflection(self):
        
        # x-axis : closed for x = 0 and x = w
        # y-axis : closed for y = 0
    
        w = self.boundary_info['width']
        
        # reflection condition
        for idx in self.particle_indices:
    
            # x-axis
            if self.r[idx,0] >= w:
                self.r[idx,0] = w
                self.u[idx,0] *= (-1) * self.gamma
                
            elif self.r[idx,0] <= 0:
                self.r[idx,0] = 0
                self.u[idx,0] *= (-1) * self.gamma
                
            # y-axis
            if self.r[idx,1] <= 0:
                self.r[idx,1] = 0
                self.u[idx,1] *= (-1) * self.gamma