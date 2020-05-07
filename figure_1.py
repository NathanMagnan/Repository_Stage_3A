import numpy as np
import scipy.optimize as opt

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
os.chdir('/home/astro/magnan')

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

## Gathering Illustris 2D plot data (subplot 1)
il = cat.Catalogue_Illustris()
il.initialise_data()
print("Illustris 2D plot data loaded")

## Gathering Illustris 2PCF data (subplot 3)
bin_min, bin_max, n_bin_2PCF = 10**(-2), 10**(3), 100
il.compute_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
min_reliable, max_reliable = 9 * 10**(-2), 1.3 * 10**(1) # according to the figure Illustris_2PCF
il.extract_reliable_2PCF(min_reliable = min_reliable, max_reliable = max_reliable)
print("Illustris 2PCF data computed")

## Gathering Illustris MST data (subplot 4)
il.compute_MST_histogram()
print("Illustris MST data computed")

## Finding the ALF parameters
count = il.count
box_size = il.box_size
Bins_reliable = il.Bins_reliable

def ALF_2PCF_reliable_log(Bins_reliable_log, alpha, beta, gamma, t0, ts):
    alf = cat.Catalogue_ALF(count = count, alpha = alpha, beta = beta, gamma = gamma, t0 = t0, ts = ts, box_size = box_size)
    alf.extract_reliable_2PCF(min_reliable = min_reliable, max_reliable = max_reliable, bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
    result = np.log10(alf.Mean_2PCF_reliable)
    print("Current ALF parameters = " , [alpha, beta, gamma, t0, ts])
    return result

Illustris_2PCF_reliable_log = np.log10(il.Mean_2PCF_reliable) # we want to fit the logarithms of the 2PCF, it will be much more precise
Illustris_Std_reliable_log = il.Std_2PCF_reliable / il.Mean_2PCF_reliable # we aproximate ln(1 + std / mean) by std / mean wich is valide as long as std << mean, and that is verified in figure Illustris_2PCF
Illustris_Std_reliable_log += 10**(-10) # to avoid the cases where an error bar is empty

p0 = [1.5, 0.45, 1.3, 0.325, 0.015]

print("Starting to compute the best ALF fit")
print("Starting ALF parameters = ", p0)
#ALF_parameters, covariance_matrix = opt.curve_fit(f = ALF_2PCF_reliable_log, xdata = np.log10(Bins_reliable), ydata = Illustris_2PCF_reliable_log, sigma = Illustris_Std_reliable_log, p0 = p0, bounds = ([0, 0, 0, 0.05, 0], [+np.inf, 1, +np.inf, 1, 0.05]), diff_step = [0.01, 0.01, 0.01, 0.001, 0.001])
print("Instead of computing the best fit we take the one found by the computing node run")
ALF_parameters = [1.53863619, 0.4500623,  1.28962183, 0.32717637, 0.0113066]
covariance_matrix = [[8.95173339e-03, -6.61400359e-03,  1.84552155e-03, -1.14351779e-04, -2.95121446e-05], [-6.61400359e-03,  4.90988434e-03, -1.24359152e-03,  9.04715272e-05, 1.31631061e-05], [1.84552155e-03, -1.24359152e-03,  4.34709760e-03,  8.01911541e-05, -1.75816276e-04], [-1.14351779e-04, 9.04715272e-05,  8.01911541e-05,  1.76781845e-05, -1.65578837e-05], [-2.95121446e-05, 1.31631061e-05, -1.75816276e-04, -1.65578837e-05, 2.07414456e-05]]
print("Best ALF fit computed")
print("Final ALF parameters = ", ALF_parameters)
print("Covariance matrix : ", covariance_matrix)

## Gathering ALF 2D plot data (subplot 2)
alf = cat.Catalogue_ALF(count = count, alpha = ALF_parameters[0], beta = ALF_parameters[1], gamma = ALF_parameters[2], t0 = ALF_parameters[3], ts = ALF_parameters[4], box_size = box_size)
alf.initialise_data()
print("ALF 2D plot data loaded")

## Gathering ALF 2PCF data (subplot 3)
alf.compute_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
alf.extract_reliable_2PCF(min_reliable = min_reliable, max_reliable = max_reliable)
print("ALF 2PCF data computed")

## Gathering ALF MST data (subplot 4)
alf.compute_MST_histogram(mode_MST = 'MultipleMST')
print("ALF MST data computed")

## Creating the entire figure
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')

fig = plt.figure(figsize = (8, 8)) # Illustris 2D plot
subplot1 = plt.subplot2grid((3, 3),(0, 0), rowspan = 3, colspan = 3)
il.plot_2D(title = "2D scatter plot of Illustris 1 galaxies", figure = subplot1)
my_file = 'Illustris_2D_plot.png'
plt.savefig(os.path.join(my_path, my_file))

fig = plt.figure(figsize = (8, 8)) # ALF 2D plot
subplot2 = plt.subplot2grid((3, 3),(0, 0), rowspan = 3, colspan = 3)
alf.plot_2D(title = "2D scatter plot of ALF galaxies", figure = subplot2)
my_file = 'ALF_2D_plot.png'
plt.savefig(os.path.join(my_path, my_file))

fig = plt.figure(figsize = (8, 8)) # ALF and Illustris 2PCFs
subplot3 = plt.subplot2grid((3, 3),(0, 0), rowspan = 3, colspan = 3)
cat.compare_2PCFs(List_catalogues = [alf, il], title = "Comparison between Illustris and ALF 2PCFs", figure = subplot3)
my_file = 'Illustris_and_ALF_2PCFs.png'
plt.savefig(os.path.join(my_path, my_file))

my_file = 'Illustris_and_ALF_MSTs.png'
my_file = os.path.join(my_path, my_file)
cat.compare_MST_histograms(List_catalogues = [alf, il], usemean = False, whichcomp = 0, title = "Comparison between Illustris and ALF MSTs", saveas = my_file)


