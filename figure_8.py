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

# Produces a plot of average autocorrelation time against temperature for MH and Wolff 

# Initialise required variables for MH tau against T:
Ts = np.linspace(0.1,5,30)
max_time = 50000
autocorr_times = []
autocorr_times_err = []

for T in Ts:

    for i in range(5):
        # Lattice must be reset in each iteration, and evolved to (beyond) equilibrium
        lattice = lat.make_lattice(30,1)
        burn = MH.evolve_and_compute_M(lattice, T**-1, 1, 0, max_time)[0]

        # evolve the lattice from equilibrium
        Ms = MH.evolve_and_compute_M(lattice, T**-1, 1, 0, max_time)[0]

        # find autocorrelation time and add to list.
        autocorr_times_i = []
        autocorr = acf.compute_autocorrelation(Ms)
        sweeps_tau_f = acf.estimate_correlation_time(autocorr)/np.size(lattice)
        autocorr_times_i.append(sweeps_tau_f)

    # Take sample mean and error
    autocorr_times.append(np.mean(autocorr_times_i))
    autocorr_times_err.append(np.std(autocorr_times_i))
    print(T)

# Save data for use in report
np.save('MH_autocorrelation_against_T', autocorr_times)
np.save('MH_autocorrelation_against_T_err', autocorr_times_err)

# Reset required variables for Wolff tau against T, smaller max time needed due to smaller autocorrelation time for Wolff:
Ts = np.linspace(0.1,5,30)
max_time = 1000
autocorr_times = []
autocorr_times_err = []

for T in Ts:

    for i in range(5):
        # Lattice must be reset after each iteration and evolved to (beyond) equilibrium
        lattice = lat.make_lattice(30,1)
        burn = W.Wolff_evolve_and_compute_M(lattice, T**-1, 1, max_time)

        # evolve the lattice from equilibrium
        Ms, sweeps = W.Wolff_evolve_and_compute_M(lattice, T**-1, 1, max_time)

        # find autocorrelation time and add to list.
        autocorr_times_i = []
        autocorr = acf.compute_autocorrelation(Ms)
        sweeps_tau_f = sweeps[acf.estimate_correlation_time(autocorr)]
        autocorr_times_i.append(sweeps_tau_f)
    
    # Take sample mean and error
    autocorr_times.append(np.mean(autocorr_times_i))
    autocorr_times_err.append(np.std(autocorr_times_i))
    print(T)

# Save data for use in report
np.save('Wolff_autocorrelation_against_T.npy', autocorr_times)
np.save('Wolff_autocorrelation_against_T_err', autocorr_times_err)