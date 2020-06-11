## Importin the libraries
import numpy as np
import mistree as mist

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## Importing the data
# getting the basepath
number_str = '00'

path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)

# gettting the data
ab.initialise_data()

print("simulation " + str(i) + " : data acquired")

## creating the data set
print("starting to work on creating the data set")
Catalogues = []

for m in range(4):
    histogram = mist.HistMST()
    histogram.setup(usenorm = False, uselog = True)
    histogram.start_group()
    Catalogues.append(histogram)

for i in range(4):
    for j in range(4):
        for k in range(4):
            
            print("starting to work on subsample " + str(i*16 + j*4 + k + 1))
            
            # determining if center, face, side or corner :
            m = 0
            if ((i == 0) or (i == 3)):
                m += 1
            if ((j == 0) or (j == 3)):
                m += 1
            if ((k == 0) or (k == 3)):
                m += 1
            
            # getting rid of the points in the small cube (i,j,k)
            lim_inf_x, lim_sup_x = 720 / 4 * i, 720 / 4 * (i + 1)
            lim_inf_y, lim_sup_y = 720 / 4 * j, 720 / 4 * (j + 1)
            lim_inf_z, lim_sup_z = 720 / 4 * k, 720 / 4 * (k + 1)
            
            CM_reduced = []
            
            for cm in ab.CM:
                if not ((cm[0] > lim_inf_x) and (cm[0] < lim_sup_x)):
                    if not ((cm[1] > lim_inf_y) and (cm[1] < lim_sup_y)):
                        if not ((cm[2] > lim_inf_z) and (cm[2] < lim_sup_z)):
                            CM_reduced.append(cm)
            
            X_reduced = np.asarray(CM_reduced[:, 0])
            Y_reduced = np.asarray(CM_reduced[:, 1])
            Z_reduced = np.asarray(CM_reduced[:, 2]) 
            
            # computing the histogram
            MST = mist.GetMST(x = X_reduced, y = Y_reduced, z = Z_reduced)
            d, l, b, s, l_index, b_index = MST.get_stats(include_index=True)
            
            # saving the catalogue
            _hist = Catalogues[m].get_hist(d, l, b, s)

for m in range(4):
    Catalogues[m].end_group()

print("data set fully created")

## Plotting the statistics
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Test_jacknife.png'
my_file = os.path.join(my_path, my_file)

Labels = ['center', 'face', 'edge', 'corner']

plot_histograms = mist.PlotHistMST()
for m in range(4):
    plot_histograms.read_mst(Catalogues[m], label = Labels[m])
plot_histograms.plot(usecomp = True, usemean = False, whichcomp = 3, figsize = (9, 6), saveas = my_file)

print("Statistics plotted")