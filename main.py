import numpy as np
import argparse
from src.SPH import SPHsolver, SPHsolverVer2

parser = argparse.ArgumentParser(description="2D SPH simulation as my own project")
parser.add_argument("--height", type = float, default = 2.0)
parser.add_argument("--width", type = float, default = 10.0)
parser.add_argument("--num_particle", type = int, default = 256)
parser.add_argument("--end_time", type = float, default = 1.0)
parser.add_argument("--radius", type = float, default = 0.0125)
parser.add_argument("--dt", type = float, default = 5e-3)
parser.add_argument("--mu", type = float, default = 1.0 * 10 **(-6))
parser.add_argument("--density", type = float, default = 1000.0)
parser.add_argument("--sound_speed", type = float, default = 256)
parser.add_argument("--g", type = float, default = 9.8)
parser.add_argument("--kn", type = int, default = 2)
parser.add_argument("--tag", type = str, default = "simulation")
parser.add_argument("--kernel_type", type = str, default = "gaussian")
parser.add_argument("--cor_type", type = str, default = "KGC")
parser.add_argument("--plot_freq", type = int, default = 256)
parser.add_argument("--plot_boundary_particle", type = bool, default = False)

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
        plot_freq = args['plot_freq'],
        plot_boundary_particle=args['plot_boundary_particle'],
        boundary_info = Boundary_Info,
        save_dir = "./result/{}.gif".format(args['tag'])
    )
    
    solver.animate()