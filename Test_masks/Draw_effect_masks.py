## Imports
import numpy as np
import pickle
import mistree as mist

import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Loading the Abacus masked histograms
Abacus_masked_hists = [0, 0, 0, 0]
Labels = ['center', 'face', 'edge', 'corner']

for m in range(4):
    my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_masks/')
    my_file = 'Masked_abacus_' + Labels[m] + '.pkl'
    my_file = os.path.join(my_path, my_file)
    
    f = open(my_file, "rb")
    Abacus_masked_hists[m] = pickle.load(f)
    f.close()

print("Abacus histograms loaded")

## Loading the Random masked histograms
Random_masked_hists = [0, 0, 0, 0]
Labels = ['center', 'face', 'edge', 'corner']

for m in range(4):
    my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_masks/')
    my_file = 'Masked_random_' + Labels[m] + '.pkl'
    my_file = os.path.join(my_path, my_file)
    
    f = open(my_file, "rb")
    Random_masked_hists[m] = pickle.load(f)
    f.close()

print("Abacus histograms loaded")

## Plotting
print("starting to plot")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Effect_masks_random.png'
my_file = os.path.join(my_path, my_file)

plot_histograms = mist.PlotHistMST()
for m in range(4):
    plot_histograms.read_mst(Random_masked_hists[m], label = Labels[m])
plot_histograms.plot(usecomp = True, usemean = False, whichcomp = 3, figsize = (9, 6), saveas = my_file)

print("Statistics plotted")