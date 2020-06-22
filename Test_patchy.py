## Imports
import numpy as np
import pandas
import pickle
import mistree as mist
import os

print("All imports successful")

## Loading the file
print("Starting to load Patchy")
my_path = '/hpcstorage/zhaoc/BOSS_PATCHY_MOCKS/PATCHY_CMASS/box1/'
my_file = 'CATALPTCICz0.466G960S997413177.dat.bz2'
my_file = os.path.join(my_path, my_file)

Patchy = pandas.read_csv(my_file, sep = ' ', names = ['X', 'Y', 'Z', 'vX', 'vY', 'vZ', 'vMax'])
X = Patchy['X'].values
Y = Patchy['Y'].values
Z = Patchy['Z'].values

print("Patchy Loaded")

## Computing Patchy MST
print("Starting to compute Patchy MST")

patchy_mst = mist.GetMST(x = X, y = Y, z = Z)
d, l, b, s = patchy_mst.get_stats(include_index = False)
patchy_histogram = mist.HistMST()
patchy_histogram.setup(usenorm = False, uselog = True)
patchy_histogram = patchy_histogram.get_hist(d, l, b, s)

print("Patchy MST computed")

## Saving the patchy MST
print("Starting to save the MST histogram")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_patchy/')
my_file = 'Test_' + '.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(patchy_histogram, f)
f.close()

print("MST histogram saved")
