import numpy as np
import importlib
import matplotlib.pylab as plt
import MH_algorithm as MH
import Wolff_algorithm as W
import init_lattice as lat
import autocorrelation_functions as acf
importlib.reload(MH)
importlib.reload(W)
importlib.reload(lat)
importlib.reload(acf)

#Produces the data for figure 

# initialise lattice
lattice = lat.make_lattice(25,0)
# evolve to eq with MH
MH_Ms, MH_sweeps = MH.evolve_and_compute_M(lattice, 1**-1, 1, 0, 100000)
MH_thermal_Ms, MH_thermal_sweeps = MH.evolve_and_compute_M(lattice, 1**-1, 1, 0, 50000)
lattice = lat.make_lattice(25,0)
Wolff_Ms, Wolff_sweeps = W.Wolff_evolve_and_compute_M(lattice, 1**-1, 1, 100)
Wolff_thermal_Ms, Wolff_thermal_sweeps = W.Wolff_evolve_and_compute_M(lattice, 1**-1, 1, 50)