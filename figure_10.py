# Firstly, import necessary modules for our program:
import numpy as np
import importlib
rng = np.random.default_rng()  
import matplotlib.pylab as plt
import time
import MH_algorithm as MH
import Wolff_algorithm as W
import init_lattice as lat
import autocorrelation_functions as acf
importlib.reload(MH)
importlib.reload(W)
importlib.reload(lat)
importlib.reload(acf)

# This program plots the 2x2 Ising case analytically as well as the Wolff and MH solutions

#First calculate analytic solution as outline in report:
J = 1
Ts_2x2 = np.linspace(0.1,5,1000)
Es_2x2 = []
T_c = 2*J/np.log(1 + np.sqrt(2)) # Critical temperature
for T in Ts_2x2:
    Es_2x2.append(-8*np.sinh(8*T**-1)/(4*np.cosh(8*T**-1)+12))

# Initialise lattices for simulation:
lattice_MH = lat.make_lattice(2,1)
lattice_Wolff = lat.make_lattice(2,1)

# Set temperatures to be evolved at:
Ts_MH = np.linspace(0.5,5,30)
Ts_Wolff = np.linspace(0.5,5,30)

# Initialise observable expectation and error arraya:
Es_MH = []
Es_MH_err = []
Es_Wolff = []
Es_Wolff_err = []

# Evolve and loop through Ts for MH:
for T in Ts_MH:
    # Burn to equilibrium 
    burn = MH.evolve_and_compute_E(lattice_MH,T**-1, J, 0, 100000)[0]
    print('Burn Finished')
    # Evolve from equilibrium
    Es = MH.evolve_and_compute_E(lattice_MH,T**-1,J,0,10000)[0]
    # Calculate batch average and error 
    Es_MH_T, Es_MH_err_T = lat.batch_average(Es)
    print(str(Es_MH_T) + ' +- ' + str(Es_MH_err_T))
    # Add to obervable arrays
    Es_MH.append(Es_MH_T)
    Es_MH_err.append(Es_MH_err_T)

print('MH Complete')

for T in Ts_Wolff:
    # Burn to equilibrium
    burn = W.Wolff_evolve_and_compute_E(lattice_Wolff,T**-1, 1, 500)[0]
    print('Burn Finished')
    # Evolve from equilibrium
    Es = W.Wolff_evolve_and_compute_E(lattice_Wolff, T**-1, 1, 750)[0]
    # Calculate batch arverage and error
    Es_Wolff_T, Es_Wolff_err_T = lat.batch_average(Es)
    print(str(Es_Wolff_T) + ' +- ' + str(Es_Wolff_err_T))
    # Add to observable arrays
    Es_Wolff.append(Es_Wolff_T)
    Es_Wolff_err.append(Es_Wolff_err_T)

#   Create and display plots:
plt.plot(Ts_2x2, Es_2x2, label = 'Analytic L = 2')
plt.plot(np.full((5,1),T_c), np.linspace(0,1,5), linestyle = '--', color = 'gray', label = r'$T_c$')
plt.errorbar(Ts_MH, Es_MH, yerr = Es_MH_err, label = 'Metropolis', capsize= 2)
plt.errorbar(Ts_Wolff, Es_Wolff, yerr = Es_Wolff_err, label = 'Wolff', capsize= 2)
plt.xlabel('T')
plt.ylabel(r'$\vert$M$\vert$')
plt.title('Analytic solutions for mean magnetisation per spin, J = ' + str(J), fontsize = 10)
plt.legend()
plt.show()

# Save data for use in report
np.save('Es_Wolff_against_T_2x2.npy', Es_Wolff)
np.save('Es_Wolff_against_T_2x2_err.npy', Es_Wolff_err)
np.save('Es_MH_against_T_2x2.npy', Es_MH)
np.save('Es_MH_against_T_2x2_err.npy', Es_MH_err)