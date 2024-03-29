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

# Produces plot of spontaneous magnetisation against temperature to compare to the onsager soln

lattice_MH = lat.make_lattice(100,1)
lattice_Wolff = lat.make_lattice(100,1)

# Set temperatures to be evolved at:
Ts_MH = np.linspace(1,5,30)
Ts_Wolff = np.linspace(1,5,30)

# Initialise needed averaging times and observable array:
Ms_MH = []
Ms_MH_err = []
Ms_Wolff = []
Ms_Wolff_err = []

# Evolve and loop through Ts:
for T in Ts_MH:
    burn = MH.evolve_and_compute_M(lattice_MH,T**-1, 1, 0, 500000)[0]
    Ms = MH.evolve_and_compute_M(lattice_MH,T**-1,1,0,100000)[0]

    Ms_MH_T, Ms_MH_err_T = lat.batch_average(Ms)
    print(Ms_MH_T)
    Ms_MH.append(Ms_MH_T)
    Ms_MH_err.append(Ms_MH_err_T)


print('MH Done')
np.save('MH_M_against_T',Ms_MH)
np.save('MH_M_against_T_err', Ms_MH_err)

for T in Ts_Wolff:
    burn = W.Wolff_evolve_and_compute_M(lattice_Wolff,T**-1, 1, 500)[0]
    print('Burn Finished')
    Ms = W.Wolff_evolve_and_compute_M(lattice_Wolff, T**-1, 1, 1000)[0]

    Ms_Wolff_T, Ms_Wolff_err_T = lat.batch_average(Ms)
    print(Ms_Wolff_T)
    Ms_Wolff.append(Ms_Wolff_T)
    Ms_Wolff_err.append(Ms_Wolff_err_T)

print('Wolff Done')
np.save('Wolff_M_against_T',Ms_Wolff)
np.save('Wolff_M_against_T_err', Ms_Wolff_err)