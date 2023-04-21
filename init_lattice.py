import numpy as np
import matplotlib as plt
import autocorrelation_functions as acf

# Produce function that produces a lattice when given desired width and initial state:
def make_lattice(width, type):
    # type takes arguments setting initial spin config:
    # +/-1 sets lattice to all be +/-1, 0 for random initial spins

    if type != 0:
        lattice = type*np.ones((width,width))
    else:
        lattice = np.random.default_rng().choice([-1,1], (width,width))

    return lattice

# Finally, produce function that graphically displays the lattice as a colourmap, using a binary colour scheme due to the binary values in question:
def show_lattice(lattice, axis, title):
    # write a function to display our output lattices as an image:
    axis.matshow(lattice, cmap=plt.cm.binary, vmin = -1, vmax = 1)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize = 10)

def near_neighbours(i, j, width):
    # using periodic BCs, must make sure spin sums correctly roll over, ie the neighbours of spin [width,width] include [0,width] 
    # etc. neighbours returned in the format (Right, Left, Up, Down)
    neighbours = [[(i+1)%width,j],[(i-1)%width,j],[i,(j-1)%width],[i,(j+1)%width]]
    return neighbours

def neighbouring_spins_sum(i, j, lattice, width):
    spin = 0
    neighbours = near_neighbours(i,j,width)
    for neighbour in neighbours:
        spin += lattice[neighbour[0]][neighbour[1]]
    return spin

def compute_magnetisation(lattice):
    # finds overall magnetisation per spin of the lattice 
    M = np.abs(np.sum(lattice)/np.size(lattice))
    return M

def compute_Chi(lattice, beta, M_Sample):
    # takes in a list of magnetisations from some sampling scheme
    # and returns susceptibility 
    return beta*np.size(lattice)*np.var(M_Sample)

def compute_E(lattice, J):
    width = len(lattice)
    E = 0
    for j in range(len(lattice)):
        for i in range(j):
            E += -J*lattice[i][j]*neighbouring_spins_sum(i,j, lattice, width)
    return  E

def batch_average(observable):
    # Takes a thermalised variable and batches the data
    tau_f = acf.estimate_correlation_time(acf.compute_autocorrelation(observable))
    print('tau_f = ' + str(tau_f))

    if tau_f == 0:
        n_batches = int(np.floor(len(observable)/2))
    else:
        n_batches = int(np.floor(len(observable)/(2*tau_f)))

    print('Data divided into ' + str(n_batches) + ' batches.')
    batch_length = 2*tau_f
    average = []
    average_square = []

    for i in range(n_batches):
        if i == n_batches - 1:
            batch = observable[i*batch_length:]
        else:
            batch = observable[i*batch_length:(i+1)*batch_length] 
        
        average.append(np.mean(batch))
        average_square.append(np.mean(np.power(batch,2)))

    average = np.mean(average)
    average_square = np.mean(average_square)
    err = np.sqrt(np.abs(((n_batches - 1)**-1)*(average_square - average**2)))
    return average, err


