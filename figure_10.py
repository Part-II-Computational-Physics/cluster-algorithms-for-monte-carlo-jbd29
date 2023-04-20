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

# Program for use in figure ten to determine the finite scaling law for the lattice

# First initialise lattice widths
lattice_widths = [2**(i + 1) for i in range(7)]

# Load measured T_c's
T_c_MH = np.load('T_c_MH.npy')
T_c_Wolff = np.load('T_c_Wolff.npy')

#Run loop for MH algorithm
MH_autocorr_time_against_width = []
MH_autocorr_time_against_width_err = []

for width in lattice_widths:
# Loop for averaging
    for i in range(5):
        # Create temporary lattice for sample measurements
        MH_temp = []

        # Initialise lattice
        lattice = lat.make_lattice(width,1)
        # Burn to equilibrium 
        burn = MH.evolve_and_compute_M(lattice, T_c_MH**-1, 1, 0, 10000)[0]
        # Evolve and find autocorrelation time
        Ms, sweeps = MH.evolve_and_compute_M(lattice,T_c_MH**-1,1,0,10000)
        MH_autocorr = acf.compute_autocorrelation(Ms)
        MH_temp.append(acf.estimate_correlation_time(MH_autocorr))

        # Repeat process for other lattice types
        lattice = lat.make_lattice(width,0)
        burn = MH.evolve_and_compute_M(lattice, T_c_MH**-1, 1, 0, 10000)[0]
        Ms, sweeps = MH.evolve_and_compute_M(lattice,T_c_MH**-1,1,0,10000)
        MH_autocorr = acf.compute_autocorrelation(Ms)
        MH_temp.append(acf.estimate_correlation_time(MH_autocorr))
        lattice = lat.make_lattice(width,-1)

        burn = MH.evolve_and_compute_M(lattice, T_c_MH**-1, 1, 0, 10000)[0]
        Ms, sweeps = MH.evolve_and_compute_M(lattice,T_c_MH**-1,1,0,10000)
        MH_autocorr = acf.compute_autocorrelation(Ms)
        MH_temp.append(acf.estimate_correlation_time(MH_autocorr))
        print(i)

    MH_autocorr_time_against_width.append(np.mean(MH_temp))
    MH_autocorr_time_against_width_err.append(np.std(MH_temp))
    print(MH_autocorr_time_against_width)

# Repeat whole process above for the Wolff algorithm
Wolff_autocorr_time_against_width = []
Wolff_autocorr_time_against_width_err = []

for width in lattice_widths:
    for i in range(5):
        Wolff_temp = []

        lattice = lat.make_lattice(width,1)
        burn = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)[0]
        Ms, sweeps = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)
        Wolff_autocorr = acf.compute_autocorrelation(Ms)
        Wolff_temp.append(acf.estimate_correlation_time(Ms))

        lattice = lat.make_lattice(width,1)
        burn = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)[0]
        Ms, sweeps = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)
        Wolff_autocorr = acf.compute_autocorrelation(Ms)
        Wolff_temp.append(acf.estimate_correlation_time(Ms))

        lattice = lat.make_lattice(width,1)
        burn = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)[0]
        Ms, sweeps = W.Wolff_evolve_and_compute_M(lattice,T_c_Wolff**-1,1,1001)
        Wolff_autocorr = acf.compute_autocorrelation(Ms)
        Wolff_temp.append(acf.estimate_correlation_time(Ms))  

    Wolff_autocorr_time_against_width.append(np.mean(Wolff_temp))
    Wolff_autocorr_time_against_width_err.append(np.std(Wolff_temp))
    print(Wolff_autocorr_time_against_width)

np.save('MH_autocorr_time_against_width.npy', MH_autocorr_time_against_width)
np.save('Wolff_autocorr_time_against_width.npy', Wolff_autocorr_time_against_width)
np.save('MH_autocorr_time_against_width_err.npy', MH_autocorr_time_against_width_err)
np.save('Wolff_autocorr_time_against_width_err.npy', Wolff_autocorr_time_against_width_err)
