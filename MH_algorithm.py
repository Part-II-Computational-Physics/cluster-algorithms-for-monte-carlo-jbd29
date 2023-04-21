import numpy as np
import matplotlib.pyplot as plt
import init_lattice as lat

# Compute Delta E for a lattice site (i,j) (14):
def compute_deltaE(i, j, lattice, J, MuH):
    # computes the change in energy for flipping a single spin at (i,j) within the lattice
    width = len(lattice)
    deltaE = 2*J*lattice[i][j]*lat.neighbouring_spins_sum(i,j,lattice,width) + 2*MuH*lattice[i][j]
    return deltaE

# Produce a single loop of the MH algorithm:
def MC_spin_flip(lattice, beta, J, MuH):
    width = len(lattice)
    # find a random lattice site
    i, j = np.random.randint(width, size = 2)
    # compute the energy if we were to flip it
    betadeltaE = beta*compute_deltaE(i,j, lattice,J,MuH)
    # use p_acc to determine if our new state is accepted:
    if betadeltaE <= 0:
        lattice[i][j] = -1*lattice[i][j] #accepted
    elif np.exp(-1*betadeltaE) >= np.random.default_rng().random():
        lattice[i][j] = -1*lattice[i][j] #accepted
    # rejected if does not satisfy either of above.

def evolve_and_show(lattice, beta, J, MuH, plot_times):
    # set up subplots for the lattice
    fig, ax = plt.subplots(1, len(plot_times), figsize = (12,4))
    sweeps = [t/np.size(lattice) for t in plot_times]
    # evolve lattice for all the times in plot times
    for t in range(plot_times[-1] + 1):
        if t>0:
            MC_spin_flip(lattice,beta,J,MuH)
        if t in plot_times:
            lat.show_lattice(lattice, ax[plot_times.index(t)], str(sweeps[plot_times.index(t)]) + ' sweeps')
    plt.show()

# Write functions to evolve and measure certain observables:
def evolve_and_compute_M(lattice, beta, J, MuH, max_time):
    # finds a time evolution of magnetisations to plot
    Ms = []
    sweeps = []
    for t in range(max_time):
        if t>0:
            MC_spin_flip(lattice, beta, J, MuH)
        Ms.append(lat.compute_magnetisation(lattice))
        sweeps.append(t/np.size(lattice))
    return Ms, sweeps

def evolve_and_compute_E(lattice, beta, J, MuH, max_time):
    # finds a time evolution of magnetisations to plot
    Es = []
    sweeps = []
    for t in range(max_time):
        if t>0:
            MC_spin_flip(lattice, beta, J, MuH)
        Es.append(lat.compute_E(lattice,J))
        sweeps.append(t/np.size(lattice))
    return Es, sweeps