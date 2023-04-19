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
# Produces data of autocorrelation against sweeps and the autocorrelation time for use in the report in figure 7

lattice = lat.make_lattice(25,1)
T = 2.8

# Evolve the lattice with Wolff
# Start by burning iterations to equilibrium
burn, burn1 = W.Wolff_evolve_and_compute_M(lattice,T**-1,1,1000)

# Now find autocorrelation
Ms, sweeps_Wolff = W.Wolff_evolve_and_compute_M(lattice,T**-1, 1, 1000)
Wolff_autocorr = acf.compute_autocorrelation(Ms)
Wolff_sweeps_tau_f = sweeps_Wolff[acf.estimate_correlation_time(Wolff_autocorr)]

# Reset lattice
lattice = lat.make_lattice(25,1)

# Evolve the lattice with MH
# Start by burning iterations to equilibrium
burn, burn1 = MH.evolve_and_compute_M(lattice, T**-1, 1, 0, 50000)
Ms, sweeps_MH = MH.evolve_and_compute_M(lattice,T**-1, 1, 0, 50000)

# Now find autocorrelation
MH_autocorr = acf.compute_autocorrelation(Ms)
MH_sweeps_tau_f = acf.estimate_correlation_time(MH_autocorr)

# Save data
np.save('MH_autocorr_evolution_sweeps.npy', sweeps_MH)
np.save('MH_autocorr_evolution_autocorr.npy', MH_autocorr)
np.save('MH_autocorr_evolution_sweeps_tau_f.npy', MH_sweeps_tau_f)
np.save('Wolff_autocorr_evolution_sweeps.npy', sweeps_Wolff)
np.save('Wolff_autocorr_evolution_autocorr.npy', Wolff_autocorr)
np.save('Wolff_auto_corr_evolution_sweeps_tau_f.npy', Wolff_sweeps_tau_f)