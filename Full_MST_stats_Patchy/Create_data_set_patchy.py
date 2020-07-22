## Imports
import numpy as np
import mistree as mist
import pickle
import pandas
from mpi4py import MPI

import sys
import os

print("All imports successful")

## Setting up the MPI
print("Starting to set up the MPI")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size() # Should be as large as possible

i = int(rank)

print("MPI set up")

## creating the data set
print("starting to work on creating the data set")

print("starting to work on simulation " + str(i))

# getting the basepath
path = '/hpcstorage/zhaoc/BOSS_PATCHY_MOCKS/PATCHY_CMASS/box1/'

# getting the list of files
list_files = os.listdir(path)

# getting the file
file = list_files[i]
my_file = os.path.join(path, file)

# reading the file
Patchy = pandas.read_csv(my_file, sep = ' ', names = ['X', 'Y', 'Z', 'vX', 'vY', 'vZ', 'vMax'])
X = Patchy['X'].values
Y = Patchy['Y'].values
Z = Patchy['Z'].values
print("file read")

# computing the MST
patchy_mst = mist.GetMST(x = X, y = Y, z = Z)
d, l, b, s = patchy_mst.get_stats(include_index = False)
print("MST computed")

# computing the histograms
patchy_histogram = mist.HistMST()
patchy_histogram.setup(usenorm = False, uselog = True)
patchy_histogram = patchy_histogram.get_hist(d, l, b, s)

# saving the MST
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Patchy/')
my_file = 'Simulation' + str(i) + '.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(patchy_histogram, f)
f.close()
print("histogram saved")

print("work done on simulation " + str(i))