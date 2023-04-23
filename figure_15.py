import numpy as np
import init_lattice as lat
import MH_algorithm as MH
import Wolff_algorithm as W
import autocorrelation_functions as acf
import importlib
import matplotlib.pyplot as plt
importlib.reload(MH)
importlib.reload(W)
importlib.reload(lat)
importlib.reload(acf)

# Produces a plot of average internal energy autocorrelation time against temperature for MH and Wolff 

# Initialise required variables for MH tau against T:
Ts = np.linspace(0.2,3.5,15)
autocorr_times_MH = []
autocorr_times_err_MH = []
T_c = 2/np.log(1 + np.sqrt(2))

for T in Ts:
    autocorr_i = []

    for i in range(10):
        # Lattice must be reset in each iteration, and evolved to (beyond) equilibrium
        lattice = lat.make_lattice(30,1)
        burn = MH.evolve_and_compute_E(lattice, T**-1, 1, 0, 10000)[0]
        # evolve the lattice from equilibrium
        Es, sweeps = MH.evolve_and_compute_E(lattice, T**-1, 1, 0, 25000)
        # find autocorrelation time and add to list.
        autocorr_i.append(acf.compute_autocorrelation(Es))
        print(i)

    # Take sample mean autocorr time
    autocorr_times_MH.append(sweeps[acf.estimate_correlation_time(np.mean(autocorr_i, axis = 0))])
    print(T)

T_c_MH_E = Ts[np.where(autocorr_times_MH == np.max(autocorr_times_MH))[0][0]]

# Save data for use in report
np.save('MH_E_autocorrelation_against_T', autocorr_times_MH)
np.save('T_c_MH_E.npy', T_c_MH_E)

# Reset required variables for Wolff tau against T, smaller max time needed due to smaller autocorrelation time for Wolff:
Ts = np.linspace(0.2,3.5,15)
max_time = 750
autocorr_times_Wolff = []
autocorr_times_err_Wolff = []

for T in Ts:
    autocorr_i = []

    for i in range(5):
        # Lattice must be reset after each iteration and evolved to (beyond) equilibrium
        lattice = lat.make_lattice(30,1)
        burn = W.Wolff_evolve_and_compute_E(lattice, T**-1, 1, max_time)[0]
        # evolve the lattice from equilibrium
        Es, sweeps = W.Wolff_evolve_and_compute_E(lattice, T**-1, 1, max_time)
        # find autocorrelation time and add to list.
        autocorr_i.append(acf.compute_autocorrelation(Es))
        print('Wolff ' + str(i))
    
    # Take sample mean autocorr time
    autocorr_times_Wolff.append(sweeps[acf.estimate_correlation_time(np.mean(autocorr_i, axis = 0))])
    print(T)

T_c_Wolff_E = Ts[np.where(autocorr_times_Wolff == np.max(autocorr_times_Wolff))[0][0]]

# Save data for use in report
np.save('Wolff_E_autocorrelation_against_T.npy', autocorr_times_Wolff)
np.save('T_c_Wolff_E.npy', T_c_Wolff_E)




