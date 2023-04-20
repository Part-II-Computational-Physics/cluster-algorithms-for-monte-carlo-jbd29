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

# Produces a plot of average autocorrelation time against temperature for MH and Wolff 

# Initialise required variables for MH tau against T:
Ts = np.linspace(0.2,3.5,15)
max_time = 50000
autocorr_times_MH = []
autocorr_times_err_MH = []
T_c = 2*J/np.log(1 + np.sqrt(2))

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
        print(i)

    # Take sample mean and error
    autocorr_times_MH.append(np.mean(autocorr_times_i))
    autocorr_times_err_MH.append(np.std(autocorr_times_i))
    print(T)

# Save data for use in report
#np.save('MH_autocorrelation_against_T', autocorr_times)
#np.save('MH_autocorrelation_against_T_err', autocorr_times_err)

# Reset required variables for Wolff tau against T, smaller max time needed due to smaller autocorrelation time for Wolff:
Ts = np.linspace(0.2,3.5,15)
max_time = 1000
autocorr_times_Wolff = []
autocorr_times_err_Wolff = []

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
        print('Wolff ' + str(i))
    
    # Take sample mean and error
    autocorr_times_Wolff.append(np.mean(autocorr_times_i))
    autocorr_times_err_Wolff.append(np.std(autocorr_times_i))
    print(T)

# Save data for use in report
#np.save('Wolff_autocorrelation_against_T.npy', autocorr_times)
#np.save('Wolff_autocorrelation_against_T_err', autocorr_times_err)


# plot the data:
plt.plot(Ts,autocorr_times_MH, label = r'$\tau_f$ estimates for Metropolis-Hastings')
# mark T_c:
plt.plot(np.full((10,1),T_c),np.linspace(0,np.max(autocorr_times_MH)+1,10), color = 'gray', linestyle = '--', label = r'$T_c$')
plt.xticks([0,1,2,3,4,5])
plt.title(r'$\tau_f$ for magnetisation against temperature on a lattice of width 50, J = 1')

plt.plot(Ts,autocorr_times_Wolff, label = r'$\tau_f$ estimates for Wolff')
plt.xticks([0,1,2,3,4,5])
plt.ylabel(r'$\tau_f$ in sweeps')
plt.xlabel('T')
plt.legend(loc = 'upper left', fontsize = 7)
plt.show()