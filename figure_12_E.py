import numpy as np
import init_lattice as lat
import MH_algorithm as MH
import Wolff_algorithm as W
import autocorrelation_functions as acf
import importlib
importlib.reload(MH)
importlib.reload(W)
importlib.reload(lat)
importlib.reload(acf)
# Produces data of internal energy autocorrelation against sweeps and the autocorrelation time for use in the report

# Initialise temperature
T = 2

# Temporary data storage
MH_autocorr_temp = []
MH_sweeps_tau_f_temp = []
Wolff_autocorr_temp = []
Wolff_sweeps_tau_f_temp = []

#Repeat and average
for i in range(5):
    print(i)
    # Reset lattice
    lattice = lat.make_lattice(25,1)
    # Start by burning iterations to equilibrium
    burn = W.Wolff_evolve_and_compute_E(lattice, T**-1, 1, 1000)[0]
    # Evolve with Wolff
    Es, sweeps_Wolff = W.Wolff_evolve_and_compute_E(lattice, T**-1, 1, 1000)
    # Now find autocorrelation
    Wolff_autocorr_temp.append(acf.compute_autocorrelation(Es))
    # Repeat with MH
    # Reset lattice
    lattice = lat.make_lattice(25,1)
    # Start by burning iterations to equilibrium
    burn = MH.evolve_and_compute_E(lattice, T**-1, 1, 0, 100000)[0]
    # Evolve the lattice with MH
    Es, sweeps_MH = MH.evolve_and_compute_E(lattice,T**-1, 1, 0, 100000)
    # Now find autocorrelation
    MH_autocorr_temp.append(acf.compute_autocorrelation(Es))

# Take Averages
MH_autocorr = np.mean(MH_autocorr_temp, axis = 0)
MH_sweeps_tau_f = sweeps_MH[acf.estimate_correlation_time(Es)]
Wolff_autocorr = np.mean(Wolff_autocorr_temp, axis = 0)
Wolff_sweeps_tau_f = acf.estimate_correlation_time(Es)

# Save data
np.save('MH_autocorr_evolution_sweeps_E.npy', sweeps_MH)
np.save('MH_autocorr_evolution_autocorr_E.npy', MH_autocorr)
np.save('MH_autocorr_evolution_sweeps_tau_f_E.npy', MH_sweeps_tau_f)
np.save('Wolff_autocorr_evolution_sweeps_E.npy', sweeps_Wolff)
np.save('Wolff_autocorr_evolution_autocorr_E.npy', Wolff_autocorr)
np.save('Wolff_auto_corr_evolution_sweeps_tau_f_E.npy', Wolff_sweeps_tau_f)