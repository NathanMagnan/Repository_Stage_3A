## Imports
import numpy as np
import mistree as mist
import pickle
from mpi4py import MPI

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## Setting up the MPI
print("Starting to set up the MPI")
comm = MPI.COMM_WORLD
n_proc = comm.Get_size() # Should be equal to 21
rank = comm.Get_rank() # Number in range(0, n_proc)

n = int(rank)

print("MPI set up")

## creating the data set
print("starting to work on creating the data set")

print("starting to work on simulation " + str(n))

# getting the basepath
if (n == 20):
    number_str = 'planck'
    path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
    path += number_str
    path += '_products/AbacusCosmos_720box_'
    path += number_str
    path += '_rockstar_halos/z0.100'
else:
    number_str = str(n)
    path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_planck_products/AbacusCosmos_720box_planck_00-'
    path += number_str
    path += '_products/AbacusCosmos_720box_planck_00-'
    path += number_str
    path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)

# gettting the data
ab.initialise_data()
CM_simu = ab.CM
n_simu = int(np.shape(CM_simu)[0])
print("data acquired")

for i in range(4):
    for j in range(4):
        for k in range(4):
            
            print("starting to work on box " + str(i*16 + j*4 + k + 1))
            
            # getting rid of the points in the small cube (i,j,k)
            lim_inf_x, lim_sup_x = 720 / 4 * i, 720 / 4 * (i + 1)
            lim_inf_y, lim_sup_y = 720 / 4 * j, 720 / 4 * (j + 1)
            lim_inf_z, lim_sup_z = 720 / 4 * k, 720 / 4 * (k + 1)
            
            # finding the nodes in the small box
            def toRemove(cm):
                if ((cm[0] > lim_inf_x) and (cm[0] < lim_sup_x)):
                    if ((cm[1] > lim_inf_y) and (cm[1] < lim_sup_y)):
                        if ((cm[2] > lim_inf_z) and (cm[2] < lim_sup_z)):
                            return(True)
                return(False)
            
            # making the smaller catalogue
            X_box, Y_box, Z_box= [], [], []
            for n in range(n_simu):
                if not (toRemove(CM_simu[n])):
                    X_box.append(CM_simu[n, 0])
                    Y_box.append(CM_simu[n, 1])
                    Z_box.append(CM_simu[n, 2])
            X_box = np.asarray(X_box)
            Y_box = np.asarray(Y_box)
            Z_box = np.asarray(Z_box)
            print("small catalogue constructed")
            
            # computing the smaller MST
            MST_box = mist.GetMST(x = X_box, y = Y_box, z = Z_box)
            d_box, l_box, b_box, s_box, l_index_box, b_index_box = MST_box.get_stats(include_index=True)
            
            # getting the histogram
            histogram_box = mist.HistMST()
            histogram_box.setup(usenorm = False, uselog = True)
            histogram_box = histogram.get_hist(d_box, l_box, b_box, s_box)
            print("histogram computed")
            
            # saving the histogram
            my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_3/')
            my_file = 'MST_stats_Simulation_' + str(n) + '_Box_' + str(i*16 + j*4 + k + 1) + '.pkl'
            my_file = os.path.join(my_path, my_file)
            
            f = open(my_file, "wb")
            pickle.dump(histogram_box, f)
            f.close()
            print("histogram saved")
            
            print("work done on box " + str(i*16 + j*4 + k + 1))

print("work done on simulation " + str(n))