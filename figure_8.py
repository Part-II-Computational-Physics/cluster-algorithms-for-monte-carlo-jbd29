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
# evolve after eq with MH
MH_thermal_Ms, MH_thermal_sweeps = MH.evolve_and_compute_M(lattice, 1**-1, 1, 0, 50000)
# reset lattice
lattice = lat.make_lattice(25,0)
# evolve to eq with Wolff
Wolff_Ms, Wolff_sweeps = W.Wolff_evolve_and_compute_M(lattice, 1**-1, 1, 100)
# evolve after eq with Wolff
Wolff_thermal_Ms, Wolff_thermal_sweeps = W.Wolff_evolve_and_compute_M(lattice, 1**-1, 1, 50)

# Save
np.save('MH_thermalisation_Ms', MH_Ms)
np.save('MH_thermalisation_sweeps', MH_sweeps)
np.save('Wolff_thermalisation_Ms', Wolff_Ms)
np.save('Wolff_thermalisation_sweeps', Wolff_sweeps)
np.save('MH_thermalised_Ms', MH_thermal_Ms)
np.save('MH_thermalised_sweeps', MH_thermal_sweeps)
np.save('Wolff_thermalised_Ms', Wolff_thermal_Ms)
np.save('Wolff_thermalised_sweeps', Wolff_thermal_sweeps)