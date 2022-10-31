import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Callable, List

class Monitor:
    def __init__(self, boundary_info : Dict, figsize = (10,6), interval : int = 10, blit : bool = True, save_dir : str = "./result/simulation.gif"):
        self.fig = plt.figure(figsize = figsize)
        self.axes = self.fig.add_subplot(xlim = (0, boundary_info['width']), ylim = (0, boundary_info['height']))
        points = self.axes.plot([],[], 'bo', ms = 20)[0]
        self.points = points
        self.frame = 0
        self.interval = interval
        self.blit = blit
        self.save_dir = save_dir
        
    def update(self, r : np.ndarray):
        self.frame += 1
        self.points.set_data(r[:,0], r[:,1])
        
    def animate(self, exec : Callable[[int], List]):
        ani = animation.FuncAnimation(self.fig, exec, interval = self.interval, blit = self.blit)
        writergif = animation.PillowWriter(fps = 32)
        ani.save(self.save_dir, writergif)