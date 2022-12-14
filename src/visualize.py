import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Callable, List, Optional, Tuple
import time

class Monitor:
    def __init__(self, n_particle : int, plot_freq : int, boundary_info : Dict, t_stats : np.array, figsize : Tuple[int,int]= (10,6), blit : bool = True, save_dir : str = "./result/simulation.gif"):
        self.fig = plt.figure(figsize = figsize)
        self.axes = self.fig.add_subplot(xlim = (0, boundary_info['width'] * 1.5), ylim = (0, boundary_info['height'] * 1.5))
        self.axes.axvline(x = boundary_info['width'], ymin = 0, ymax = boundary_info['height'])
        self.points = self.axes.plot([],[], 'bo', ms = 10)[0]
        self.boundary = self.axes.plot([],[],'ko', ms = 10)[0]
        self.frame = 0
        self.t_stats = t_stats
        self.blit = blit
        self.save_dir = save_dir
        self.plot_freq = plot_freq
        self.boundary_info = boundary_info
        self.n_particle = n_particle
        self.time_text = self.axes.text(0.9, 0.9, s = "", fontsize = 12, transform = self.axes.transAxes)
        
    def update(self, r : np.ndarray, boundary : Optional[np.ndarray] = None):
        self.frame += 1
        
        if self.frame >= len(self.t_stats):
            pass
        else:
            self.points.set_data(r[:,0], r[:,1])
            self.time_text.set_text("t={:.3f}".format(self.t_stats[self.frame-1]))
        
        if boundary is not None:
            self.boundary.set_data(boundary[:,0], boundary[:,1])
        
    def animate(self, exec : Callable[[int], List]):
        
        start_time = time.time()
        
        ani = animation.FuncAnimation(self.fig, exec, frames = self.t_stats, blit = self.blit)
        writergif = animation.PillowWriter(fps = self.plot_freq)
        ani.save(self.save_dir, writergif)
        
        end_time = time.time()
        
        print("| computation clear | condition : (h:{:.2f}, w:{:.2f}, t:{:.2f}, num particle:{}) | run time : {:.2f} |".format(
            self.boundary_info['height'], self.boundary_info['width'], self.t_stats[-1], self.n_particle, end_time - start_time
        ))