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

## Gathering Illustris 2PCF data (subplot 3)
bin_min, bin_max, n_bin_2PCF = 10**(-2), 10**(3), 100
il.compute_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
min_reliable, max_reliable = 9 * 10**(-2), 1.3 * 10**(1) # according to the figure Illustris_2PCF
il.compute_reliable_2PCF(min_realiable = min_reliable, max_reliable = max_reliable)

## Gathering Illustris MST data (subplot 4)
il.compute_MST_histogram()

## Finding the ALF parameters
count = il.count
box_size = il.box_size
Bins_reliable = il.Bins_reliable

def ALF_2PCF_reliable_log(Bins_reliable, alpha, beta, gamma, t0, ts):
    alf = cat.Catalogue_ALF(count = count, alpha = alpha, beta = beta, gamma = gamma, t0 = t0, ts = ts, box_size = box_size)
    alf.extract_reliable_2PCF(min_reliable = min_reliable, max_reliable = max_reliable, bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
    result = np.log10(alf.Mean_2PCF_reliable)
    return result

Illustris_2PCF_reliable_log = np.log10(il.Mean_2PCF_reliable) # we want to fit the logarithms of the 2PCF, it will be much more precise
Illustris_Std_reliable_log = il.Std_2PCF_reliable / il.Mean_2PCF_reliable # we aproximate ln(1 + std / mean) by std / mean wich is valide as long as std << mean, and that is verified in figure Illustris_2PCF

ALF_parameters, covariance_matrix = opt.curve_fit(f = ALF_2PCF_reliable_log, xdata = Bins_reliable, ydata = Illustris_2PCF_reliable_log, sigma = Illustris_Std_reliable_log, p0 = [1.5, 0.5, 1.3, 0.3, 0.01], bounds = ([0, 0, 0, 0.05, 0], [+np.inf, 1, +np.inf, 1, 0.05]))

print("ALF parameters = ", ALF_parameters)
print("Covariance matrix : ", covariance_matrix)

## Gathering ALF 2D plot data (subplot 2)
alf = cat.Catalogue_ALF(count = count, alpha = ALF_parameters[0], beta = ALF_parameters[1], gamma = ALF_parameters[2], t0 = ALF_parameters[3], ts = ALF_parameters[4], box_size = box_size)
alf.initialise_data()

## Gathering ALF 2PCF data (subplot 3)
alf.compute_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
alf.compute_reliable_2PCF(min_realiable = min_reliable, max_reliable = max_reliable)

## Gathering ALF MST data (subplot 4)
alf.compute_MST_histogram(mode_MST = 'MultipleMST')

## Creating the entire figure
fig = plt.figure()

subplot1 = plt.subplot2grid((9,9),(0,0), rowspan = 3, colspan = 3) # Illustris 2D plot
subplot2 = plt.subplot2grid((9,9),(0,3), rowspan = 3, colspan = 3) # ALF 2D plot
subplot3 = plt.subplot2grid((9,9),(0,6), rowspan = 3, colspan = 3) # Illustris and ALF 2PCFs
subplot4 = plt.subplot2grid((9,9),(3,0), rowspan = 6, colspan = 9) # Illustris and ALF MST histograms

# I need to find how to get my figure inside the subplots
subplot1.plot(figure = il.plot_2D(title = "2D scatter plot of Illustris 1 galaxies"))
subplot2.plot(figure = alf.plot_2D(title = "2D scatter plot of ALF galaxies"))
subplot3.plot(figure = cat.compare_2PCFs(List_catalogues = [alf, il], title = "Comparison between Illustris and ALF 2PCFs"))
subplot4.plot(figure = cat.compare_MST_histograms(List_catalogues = [alf, il], title = "Comparison between Illustris and ALF MSTs"))

plt.show(block = True)