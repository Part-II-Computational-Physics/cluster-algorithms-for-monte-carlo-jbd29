import numpy as np
import Wolff_algorithm as W
import MH_algorithm as MH
import time
import init_lattice as lat

# Finds temperature and lattice size complexity of single iterations

Ts = np.linspace(0.5, 5, 50)
Wolff_execute_time = []
MH_execute_time = []
MH_execute_time_err = []
Wolff_execute_time_err = []

for T in Ts:
    # Set up temmporary data storage
    Wolff_execute_time_i = []
    MH_execute_time_i = []

    # Average over loops
    for i in range(20):
        lattice = lat.make_lattice(100, 1)
        start = time.time()
        W.Wolff_spin_flip(lattice, T**-1, 1)
        end = time.time()
        Wolff_execute_time_i.append(end-start)
        lattice = lat.make_lattice(100, 1)
        start = time.time()
        MH.MC_spin_flip(lattice, T**-1, 1, 0)
        end = time.time()
        MH_execute_time_i.append(end - start)
    
    # Take mean and error
    MH_execute_time.append(np.mean(MH_execute_time_i))
    MH_execute_time_err.append(np.std(MH_execute_time_i)/np.sqrt(len(MH_execute_time_i)))
    Wolff_execute_time.append(np.mean(Wolff_execute_time_i))
    Wolff_execute_time_err.append(np.std(Wolff_execute_time_i)/np.sqrt(len(Wolff_execute_time_i)))       

# Save for use in report
np.save('MH_execute_time_against_T.npy', MH_execute_time)
np.save('MH_execution_time_against_T_err.npy', MH_execute_time_err)
np.save('Wolff_execute_time_against_T.npy', Wolff_execute_time)
np.save('Wolff_execute_time_against_T_err.npy', Wolff_execute_time_err)



#Lattice size complexity:

lattice_widths = np.arange(25,500,25)
Wolff_execute_time = []
MH_execute_time = []
MH_execute_time_err = []
Wolff_execute_time_err = []

for width in lattice_widths:
    # Set up temmporary data storage
    Wolff_execute_time_i = []
    MH_execute_time_i = []

    # Average over loops
    for i in range(5):
        lattice = lat.make_lattice(width, 1)
        start = time.time()
        W.Wolff_spin_flip(lattice, 1, 1)
        end = time.time()
        Wolff_execute_time_i.append(end-start)
        lattice = lat.make_lattice(width, 1)
        start = time.time()
        MH.MC_spin_flip(lattice, 1, 1, 0)
        end = time.time()
        MH_execute_time_i.append(end - start)
    
    # Take mean and error
    MH_execute_time.append(np.mean(MH_execute_time_i))
    MH_execute_time_err.append(np.std(MH_execute_time_i)/np.sqrt(len(MH_execute_time_i)))
    Wolff_execute_time.append(np.mean(Wolff_execute_time_i))
    Wolff_execute_time_err.append(np.std(Wolff_execute_time_i)/np.sqrt(len(Wolff_execute_time_i)))   

# Save for use in report
np.save('MH_execute_time_against_T.npy', MH_execute_time)
np.save('MH_execution_time_against_T_err.npy', MH_execute_time_err)
np.save('Wolff_execute_time_against_T.npy', Wolff_execute_time)
np.save('Wolff_execute_time_against_T_err.npy', Wolff_execute_time_err)