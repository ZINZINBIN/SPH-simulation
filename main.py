import numpy as np
import argparse
from src.SPH import SPHsolver

parser = argparse.ArgumentParser(description="2D SPH simulation as my own project")
parser.add_argument("--height", type = float, default = 2.0)
parser.add_argument("--width", type = float, default = 10.0)
parser.add_argument("--num_particle", type = int, default = 1024)
parser.add_argument("--end_time", type = float, default = 10.0)
parser.add_argument("--radius", type = float, default = 0.0125)
parser.add_argument("--dt", type = float, default = 1e-2)
parser.add_argument("--mu", type = float, default = 1.0 * 10 **(-1))
parser.add_argument("--density", type = float, default = 1000.0)
parser.add_argument("--sound_speed", type = float, default = 256)
parser.add_argument("--g", type = float, default = 9.8)
parser.add_argument("--kn", type = int, default = 2)
parser.add_argument("--tag", type = str, default = "simulation")
parser.add_argument("--kernel_type", type = str, default = "gaussian")
parser.add_argument("--cor_type", type = str, default = "KGC")
parser.add_argument("--use_bp", type = bool, default = False)
parser.add_argument("--plot_freq", type = int, default = 1024)
parser.add_argument("--plot_boundary_particle", type = bool, default = False)
parser.add_argument("--verbose", type = bool, default = True)

args = vars(parser.parse_args())

if __name__ == "__main__":
    
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