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
n_proc = comm.Get_size() # Should be equal to 3 * 22
rank = comm.Get_rank() # Number in range(0, n_proc)

n = int(rank)

print("MPI set up")

## Getting i, j from the rank

if (n >= 44):
    i = n - 44
    j = 2
elif (n >= 22):
    i = n - 22
    j = 1
else:
    i = n
    j = 0

## creating the data set
print("starting to work on creating the data set")

print("starting to work on simulation " + str(i))

# getting the basepath
if (i < 20):
    number_str = str(i)
    path = '/hpcstorage/magnan/Abacus_720/AbacusCosmos_720box_planck_products/AbacusCosmos_720box_planck_00-'
    path += number_str
    path += '_products/AbacusCosmos_720box_planck_00-'
    path += number_str
    path += '_rockstar_halos/z0.100'
elif (i == 20):
    number_str = 'planck'
    path = '/hpcstorage/magnan/Abacus_720/AbacusCosmos_720box_products/AbacusCosmos_720box_'
    path += number_str
    path += '_products/AbacusCosmos_720box_'
    path += number_str
    path += '_rockstar_halos/z0.100'
else:
    number_str = str(0) + str(5)
    path = '/hpcstorage/magnan/Abacus_720/AbacusCosmos_720box_products/AbacusCosmos_720box_'
    path += number_str
    path += '_products/AbacusCosmos_720box_'
    path += number_str
    path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)
print("simulation " + str(i) + " : catalogue created")

# gettting the data
ab.initialise_data()
print("simulation " + str(i) + " : data acquired")

# reducing the data set
density = 6214031 / (2500**3)
n_haloes = int(density * 720**3)

if (j == 0): # we reduce via a Mass Cut
    Masses = ab.Masses.copy()
    CM = ab.CM.copy()
    
    Masses_sorted = np.sort(Masses)
    mass_cut = Masses_sorted[- n_haloes]
    
    CM_reduced = []
    for k in range(np.shape(Masses)[0]):
        if (Masses[k] >= mass_cut):
            CM_reduced.append(CM[k])
    ab.CM = np.asarray(CM_reduced)
    
elif (j == 1): # we reduce via a Random Cut
    CM = ab.CM.copy()
    CM_reduced = CM[np.random.choice(np.shape(CM)[0], n_haloes, replace = False), :]
    ab.CM = CM_reduced

print("simulation " + str(i) + " : distribution reduced to BigMD density")

# computing the histogram
ab.compute_MST_histogram(jacknife = False)
print("simulation " + str(i) + " : histogram computed")

# Saving the data
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Abacus_vs_BigMD')
if (j == 0):
    my_file = 'MST_Mass_cut_' + str(i) + '.pkl'
elif (j == 1):
    my_file = 'MST_random_cut' + str(i) + '.pkl'
else:
    my_file = 'MST_full' + str(i) + '.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(ab.MST_histogram, f)
f.close()
print("simulation " + str(i) + " : histogram saved")