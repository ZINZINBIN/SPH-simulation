import numpy as np
import time
from typing import Dict, Optional
from src.kernels import compute_kernel
from src.NNPS import NNPS
from src.correct import KGC
from src.force import compute_pressure, compute_du_viscous, compute_du_pressure, compute_du_gravity
from src.visualize import Monitor
from tqdm.auto import tqdm

# For 2D SPH solver
class SPHsolver:
    def __init__(
        self,  
        Np : int, 
        ti : float = 0,
        tf : float = 4.0,
        dt : float = 1e-3,
        m : float = 1.0,
        mu : float = 0.1,
        rho : float = 1.0,
        C : float = 100,
        g : float = 9.8,
        radius : float = 0.05,
        kn : int = 2,
        kernel_type : str = 'gaussian',
        cor_type : str = 'KGC',
        plot_freq : int = 10,
        boundary_info : Optional[Dict] = None,
        ):
        
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
        self.m = np.ones((self.Nt,)) * m
        self.u = np.zeros_like(self.r)
        self.du_dt = np.zeros_like(self.r)
        self.C = C
        self.g = g
        
        # video configuration
        self.plot_freq = plot_freq
        self.monitor = Monitor(boundary_info, figsize = (10,6), interval = 10, blit = True)
        
    def animate(self):
        self.monitor.animate(self.update)
        
    def initialize_boundary(self):
        
        h = self.boundary_info['height']
        w = self.boundary_info['width']
        dims = self.boundary_info['dims']
        
        self.Nb = int(h/self.vol)*2 + int(w/self.vol)
        self.Nt = self.Np + self.Nb
        
        self.r = np.zeros((self.Nt, dims))
        self.ptype = np.ones((self.Nt,1)).reshape(-1,)
        
        for idx in range(0,int(h/self.vol)):
            self.r[idx,0] = -self.radius
            self.r[idx,1] = self.radius * (2 * idx + 1)
            self.ptype[idx] = 0
        
        for idx in range(0, int(w/self.vol)):
            self.r[idx + int(h/self.vol),0] = self.radius * (2 * idx + 1)
            self.r[idx + int(h/self.vol),1] = - self.radius
            self.ptype[idx + int(h/self.vol)] = 0
            
        for idx in range(0,int(h/self.vol)):
            self.r[idx + int(h/self.vol) + int(w/self.vol),1] = self.radius * (2 * idx + 1)
            self.r[idx + int(h/self.vol) + int(w/self.vol),0] = h + self.radius
            self.ptype[idx + int(h/self.vol) + int(w/self.vol)] = 0
    
    def initialize_particles(self):
        
        h = self.radius
        w = self.radius
        
        for idx in range(self.Nt):
            
            if self.ptype[idx] == 1:
                h += self.radius * 2
                self.r[idx,:] = np.array([w,h])
                
            if h >= self.boundary_info['height']:
                h = self.radius
                w += self.radius * 2
    
    def update(self, idx : int):
        # mass conservation
        self.update_mass()
            
        # boundary condition
        self.update_boundary() 
        
        # momentum condition
        self.update_momentum()
        
        # animation
        self.monitor.update(self.r)
        
        return self.monitor.points,
        
    def solve(self):
        
        ti = self.ti
        tf = self.tf
        dt = self.dt
        
        steps = round((tf-ti)/dt)
        ts = np.linspace(ti, tf, steps)
        
        count = 0
        
        for t in tqdm(ts):
            # mass conservation
            self.update_mass()
                
            # boundary condition
            self.update_boundary() 
            
            # momentum condition
            self.update_momentum()
            
            if count % self.plot_freq == 0:
                print("t = {:.6f}, umax = {:.3f}".format(t, np.max(np.linalg.norm(self.u, axis = 1))))
                
            count += 1
            
            # animation
            self.monitor.update(self.r)
    
    def update_mass(self):
        rho = np.zeros_like(self.rho)
        drho_dt = np.zeros_like(self.rho)
        P = np.zeros_like(self.P)
        
        for idx in range(0,self.Nt):
            adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
            W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
            # W_Wd, dW_Wd = KGC(idx, adj_idx, self.r, W_Wd, dW_Wd, self.m, self.rho)
            
            drho_dt[idx] = (-1) * self.rho[idx] * np.sum(self.m[adj_idx] / self.rho[adj_idx] * np.sum(dW_Wd * (self.u[adj_idx] - self.u[idx]), axis = 1), axis = 0)
            
        rho = self.rho + self.dt * drho_dt * (self.ptype == 1)
        P = compute_pressure(rho, self.rho0, self.C)
        
        self.rho = rho
        self.P = P
    
    def update_boundary(self):
        # No-Penetration condition / Neumann Boundary condition
        for idx in range(0,self.Nt):
            if self.ptype[idx] <= 0:
                adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
                W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
                filter = np.sum(self.m[adj_idx] / self.rho[adj_idx] * W_Wd * (self.ptype[adj_idx] == 1), axis = 0)
                
                if filter >= 0.01:
                    m_rho = self.m[adj_idx] / self.rho[adj_idx]
                    m_rho = m_rho.reshape(-1,1)
                    is_particle = (self.ptype[adj_idx] == 1).reshape(-1,1)
                    
                    self.u[idx, :] = (-1) * np.sum(m_rho * self.u[adj_idx] * W_Wd.reshape(-1,1) * is_particle, axis = 0) / filter
                    
                    self.P[idx] = np.sum(self.P[adj_idx] * self.m[adj_idx] / self.rho[adj_idx] * W_Wd.reshape(-1,1) * is_particle) / filter + np.sum(self.rho[adj_idx] * (self.r[adj_idx, 1] - self.r[idx, 1]) * self.g)
                    self.rho[idx] = self.P[idx] / self.C**2 + self.rho0[idx]
    
    def update_momentum(self):
        
        for idx in range(0,self.Nt):
            adj_idx = NNPS(self.r, self.r[idx], self.r_sup)
            W_Wd, dW_Wd = compute_kernel((-1) * (self.r[adj_idx] - self.r[idx]), self.r_sup, self.kernel_type)
            # W_Wd, dW_Wd = KGC(idx, adj_idx, self.r, W_Wd, dW_Wd, self.m, self.rho)
            
            du_dt = compute_du_pressure(self.rho, self.m, self.C, idx, adj_idx, dW_Wd) + compute_du_viscous(self.rho, self.mu, self.r, self.m, self.u, idx, adj_idx, dW_Wd) + compute_du_gravity(self.g)
            
            self.u[idx,:] = self.u[idx,:] + self.dt * du_dt * (self.ptype[idx] == 1).reshape(-1,1)
            self.r[idx,:] = self.r[idx,:] + self.dt * self.u[idx,:] * (self.ptype[idx] == 1).reshape(-1,1)
     
    def correct_kernel(self):
        pass