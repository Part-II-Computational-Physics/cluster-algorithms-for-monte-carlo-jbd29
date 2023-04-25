import numpy as np
import init_lattice as lat
import MH_algorithm as MH
import Wolff_algorithm as W
import autocorrelation_functions as acf
import statsmodels.api as sm
import importlib
importlib.reload(MH)
importlib.reload(W)
importlib.reload(lat)
importlib.reload(acf)

# Program for use in report to determine the finite scaling law for the lattice

# First initialise lattice widths
lattice_widths = [2,5,10,50,100]

# Load measured T_c's
T_c_MH = 2/np.log(1 + np.sqrt(2))
T_c_Wolff = 2/np.log(1 + np.sqrt(2))

#Run loop for MH algorithm
MH_autocorr_time_against_width = []

for width in lattice_widths:

    # Temp data storage
    MH_autocorr = []
    # Loop for averaging
    for i in range(5):

        # Initialise lattice
        lattice = lat.make_lattice(width,1)
        # Burn to equilibrium 
        burn = MH.evolve_and_compute_M(lattice, T_c_MH**-1, 1, 0, 10000)[0]
        # Evolve and find autocorrelation time
        Ms = MH.evolve_and_compute_M(lattice, T_c_MH**-1, 1, 0, 50000)[0]

        # Batch up the data to get more measurements:
        for j in range(5):
            Ms_sample = np.array_split(Ms, 5)[j]
            MH_autocorr.append(acf.compute_autocorrelation(Ms_sample))

    print(i)
    MH_autocorr_time_against_width.append(acf.estimate_correlation_time(np.mean(MH_autocorr, axis = 0)))
    print('Width = ' + str(width) + ', autocorr = ' + str(MH_autocorr_time_against_width))

# Repeat whole process above for the Wolff algorithm
Wolff_autocorr_time_against_width = []

for width in lattice_widths:

    # Temp data storage
    Wolff_autocorr = []
    for i in range(5):

        lattice = lat.make_lattice(width, 1)
        burn = W.Wolff_evolve_and_compute_M(lattice, T_c_Wolff**-1, 1, 1000)[0]
        print('burn done')
        Ms = W.Wolff_evolve_and_compute_M(lattice, T_c_Wolff**-1, 1, 2000)[0]

        for j in range(5):
            Ms_sample = np.array_split(Ms, 5)[j]
            Wolff_autocorr.append(acf.compute_autocorrelation(Ms_sample))

    print(i)
    Wolff_autocorr_time_against_width.append(acf.estimate_correlation_time(np.mean(Wolff_autocorr, axis = 0)))
    print('Width = ' + str(width) + ', Wolff autocorr = ' + str(Wolff_autocorr_time_against_width))

np.save('MH_autocorr_time_against_width._analytic.npy', MH_autocorr_time_against_width)
np.save('Wolff_autocorr_time_against_width_analytic.npy', Wolff_autocorr_time_against_width)

# Now use statsmodels to fit the data to linear regression:
# Firstly take logs:
lnL = np.log(lattice_widths)
lnMH_autocorr_time_against_width = np.log(MH_autocorr_time_against_width)
lnWolff_autocorr_time_against_width = np.log(Wolff_autocorr_time_against_width)

# Linear regression for MH
A = np.vstack([lnL, np.ones(len(lnMH_autocorr_time_against_width))]).T
model = sm.OLS(lnMH_autocorr_time_against_width, A)
results = model.fit()
m , c = results.params
m_err, c_err = results.bse
R2 = results.rsquared

# Save for use in report
np.save('MH_autocorr_time_against_width_params_analytic.npy', [m, c, R2])
np.save('MH_autocorr_time_against_width_params_err_analytic.npy', [m_err, c_err])

# Linear regression for Wolff
A = np.vstack([lnL, np.ones(len(lnWolff_autocorr_time_against_width))]).T
model = sm.OLS(lnWolff_autocorr_time_against_width, A)
results = model.fit()
m , c = results.params
m_err, c_err = results.bse
R2 = results.rsquared

# Save for use in report 
np.save('Wolff_autocorr_time_against_width_params_analytic.npy', [m, c, R2])
np.save('Wolff_autocorr_time_against_width_params_err_analytic.npy', [m_err, c_err])
