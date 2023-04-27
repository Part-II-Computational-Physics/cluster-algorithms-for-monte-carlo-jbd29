import numpy as np
import init_lattice as lat
import Wolff_algorithm as W

# Plots cluster size against temperature
Ts = np.linspace(0.5, 3, 30)
cluster_size_against_T = []
cluster_size_against_T_err = []

for T in Ts:
    lattice = lat.make_lattice(25, 1)
    for i in range(1000):
        temp = []
        temp.append(W.Wolff_spin_flip(lattice, T**-1, 1))
    cluster_size_against_T.append(np.mean(temp))
    print(np.mean(temp), T)
    cluster_size_against_T_err.append(np.std(temp))

# Save for use in report
np.save('cluster_size_against_T.npy', cluster_size_against_T)
np.save('cluster_size_against_T_err.npy', cluster_size_against_T_err)