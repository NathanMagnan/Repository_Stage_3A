## Imports
import numpy as np
import pandas
import pickle
import mistree as mist
import os

print("All imports successful")

## Loading the file
print("Starting to load BigMD")
my_path = '/hpcstorage/zhaoc/BOSS_PATCHY_MOCKS/ref/'
my_file = 'Box_HAM_z0.465600_nbar3.976980e-04_scat0.2384.dat'
my_file = os.path.join(my_path, my_file)

BigMD = pandas.read_csv(my_file, sep = ' ', names = ['X', '0', 'Y', '1', 'Z', '2', 'vX', '3', 'vY', '4', 'vZ', '5', 'vMax', '6', 'Other'])
X = BigMD['X'].values
Y = BigMD['Y'].values
Z = BigMD['Z'].values

print("BigMD Loaded")

## Computing BigMD MST
print("Starting to compute BigMD MST")

BigMD_mst = mist.GetMST(x = X, y = Y, z = Z)
d, l, b, s = BigMD_mst.get_stats(include_index = False)
BigMD_histogram = mist.HistMST()
BigMD_histogram.setup(usenorm = False, uselog = True)
BigMD_histogram = BigMD_histogram.get_hist(d, l, b, s)

print("BigMD MST computed")

## Saving the BigMD MST
print("Starting to save the MST histogram")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_patchy/')
my_file = 'BigMD' + '.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(BigMD_histogram, f)
f.close()

print("MST histogram saved")
