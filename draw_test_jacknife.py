## Importing the libraries
import numpy as np
import mistree as mist
import pickle

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")
## Plotting the statistics
MST_dicts = [0, 0, 0, 0]
Labels = ['center', 'face', 'edge', 'corner']

for m in range(4):
	my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_jacknife/')
	my_file = 'Test_3_' + Labels[m] + '.pkl'
	my_file = os.path.join(my_path, my_file)
	
	f = open(my_file, "rb")
	MST_dicts[m] = pickle.load(f)
	f.close()


my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Test_jacknife_3.png'
my_file = os.path.join(my_path, my_file)

plot_histograms = mist.PlotHistMST()
for m in range(4):
    plot_histograms.read_mst(MST_dicts[m], label = Labels[m])
plot_histograms.plot(usecomp = True, usemean = False, whichcomp = 3, figsize = (9, 6), saveas = my_file)

print("Statistics plotted")
