import init_lattice as lat
import numpy as np
import matplotlib.pyplot as plt

## Implementation of the Wolff algorithm for the ising model for use in the report.

def Wolff_spin_flip(lattice, beta, J):
    width = len(lattice)
    # find a random lattice site as a seed
    i, j = np.random.randint(width, size = 2)
    seed = [i,j]
    # find its spin to be added to cluster
    sigma = lattice[i][j]
    lattice[i][j] = -1*lattice[i][j]
    cluster_size = 1
    unvisited = [seed]
    # calculate the addition probability for like spins
    p_add = 1 - np.exp(-2*beta*J)
    while (len(unvisited)>0):
        #remove last unvisited sited and assign it as the site
        site = unvisited.pop()
        for neighbour in lat.near_neighbours(site[0],site[1],width):
            if lattice[neighbour[0],neighbour[1]] == sigma and p_add > np.random.default_rng().random():
                # successful addition to the cluster so flip spin:
                lattice[neighbour[0],neighbour[1]] = -1*lattice[neighbour[0],neighbour[1]]
                # as successfull, add to unvisited neighbours, and update cluster size:
                unvisited.append([neighbour[0],neighbour[1]])
                cluster_size += 1
    # Return the size of the cluster for analysis
    return cluster_size

# evolve the lattice with Wolff and show it:
def Wolff_evolve_and_show(lattice, beta, J, plot_times):
    fig, ax = plt.subplots(1, len(plot_times), figsize = (12,4))
    sweeps = [0]
    for t in range(plot_times[-1]+1):
        if t>0:
            cluster_size = Wolff_spin_flip(lattice, beta, J)
            sweeps_in_iteration = cluster_size/np.size(lattice)
            sweeps.append(sweeps[-1]+sweeps_in_iteration)
        if t in plot_times:
            lat.show_lattice(lattice, ax[plot_times.index(t)], str(round(sweeps[t],3)) + ' sweeps')
    
    plt.show()

# evolve the lattice with Wolff and return a magnetisation time series
def Wolff_evolve_and_compute_M(lattice, beta, J, max_time):
    Ms = []
    sweeps = [0]
    for t in range(max_time):
        if t>0:
            cluster_size = Wolff_spin_flip(lattice, beta, J)
            sweeps_in_iteration = cluster_size/np.size(lattice)
            sweeps.append(sweeps[-1]+sweeps_in_iteration)
        Ms.append(np.abs(lat.compute_magnetisation(lattice)))
    return Ms, sweeps

def Wolff_evolve_and_compute_E(lattice, beta, J, max_time):
    Es = []
    sweeps = [0]
    for t in range(max_time):
        if t>0:
            cluster_size = Wolff_spin_flip(lattice, beta, J)
            sweeps_in_iteration = cluster_size/np.size(lattice)
            sweeps.append(sweeps[-1]+sweeps_in_iteration)
        Es.append(lat.compute_E(lattice,J))
    return Es, sweeps