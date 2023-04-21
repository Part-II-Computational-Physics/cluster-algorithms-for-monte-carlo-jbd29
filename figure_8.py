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

#Produces the data for figure 8

# initialise lattice
lattice = lat.make_lattice(25,0)
# evolve to eq with MH
MH_Es, MH_sweeps = MH.evolve_and_compute_E(lattice, 1**-1, 1, 0, 100000)
print('initial MH done')
# evolve after eq with MH
MH_thermal_Es, MH_thermal_sweeps = MH.evolve_and_compute_E(lattice, 1**-1, 1, 0, 50000)
print('thermalised MH done')
# reset lattice
lattice = lat.make_lattice(25,0)
# evolve to eq with Wolff
Wolff_Es, Wolff_sweeps = W.Wolff_evolve_and_compute_E(lattice, 1**-1, 1, 100)
print('intitial Wolff done')
# evolve after eq with Wolff
Wolff_thermal_Es, Wolff_thermal_sweeps = W.Wolff_evolve_and_compute_E(lattice, 1**-1, 1, 50)

# Save
np.save('MH_thermalisation_Es', MH_Es)
np.save('MH_thermalisation_sweeps_Es', MH_sweeps)
np.save('Wolff_thermalisation_Es', Wolff_Es)
np.save('Wolff_thermalisation_sweeps_Es', Wolff_sweeps)
np.save('MH_thermalised_Es', MH_thermal_Es)
np.save('MH_thermalised_sweeps_Es', MH_thermal_sweeps)
np.save('Wolff_thermalised_Es', Wolff_thermal_Es)
np.save('Wolff_thermalised_sweeps_Es', Wolff_thermal_sweeps)