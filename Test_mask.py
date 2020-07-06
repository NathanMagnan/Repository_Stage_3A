## Imports
import numpy as np
import mistree as mist
import pandas

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## MST Abacus
print("starting to work on the abacus simulation")

number_str = '00'
path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'

# first without systematics
ab = cat.Catalogue_Abacus(basePath = path)
ab.initialise_data()

n_abacus = int(np.shape(ab.CM)[0])

density = np.shape(ab.CM)[0] / (720**3)
print(density)

ab.compute_MST_histogram(jacknife = False)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Full_Abacus"

np.savetxt(str(target) + str(i) + "_X_d", ab.MST_histogram['x_d'])
np.savetxt(str(target) + str(i) + "_Y_d", ab.MST_histogram['y_d'])
np.savetxt(str(target) + str(i) + "_X_l", ab.MST_histogram['x_l'])
np.savetxt(str(target) + str(i) + "_Y_l", ab.MST_histogram['y_l'])
np.savetxt(str(target) + str(i) + "_X_b", ab.MST_histogram['x_b'])
np.savetxt(str(target) + str(i) + "_Y_b", ab.MST_histogram['y_b'])
np.savetxt(str(target) + str(i) + "_X_s", ab.MST_histogram['x_s'])
np.savetxt(str(target) + str(i) + "_Y_s", ab.MST_histogram['y_s'])

# then with systematics
ab = cat.Catalogue_Abacus(basePath = path)
ab.initialise_data()

CM_new = []
for cm in ab.CM:
    if (((cm[0] - 360)**2 + (cm[1] - 360)**2 + (cm[2] - 360)**2) < 1600):
        CM_new.append(CM.copy())
ab.CM = np.reshape(CM_new, (-1, 3))

density = np.shape(ab.CM)[0] / (720**3)
print(density)

ab.compute_MST_histogram(jacknife = False)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Masked_Abacus"

np.savetxt(str(target) + str(i) + "_X_d", ab.MST_histogram['x_d'])
np.savetxt(str(target) + str(i) + "_Y_d", ab.MST_histogram['y_d'])
np.savetxt(str(target) + str(i) + "_X_l", ab.MST_histogram['x_l'])
np.savetxt(str(target) + str(i) + "_Y_l", ab.MST_histogram['y_l'])
np.savetxt(str(target) + str(i) + "_X_b", ab.MST_histogram['x_b'])
np.savetxt(str(target) + str(i) + "_Y_b", ab.MST_histogram['y_b'])
np.savetxt(str(target) + str(i) + "_X_s", ab.MST_histogram['x_s'])
np.savetxt(str(target) + str(i) + "_Y_s", ab.MST_histogram['y_s'])

## MST Random
print("starting to work on the random catalogue")

# first without systematics
X = np.random.random(n_abacus) * 720
Y = np.random.random(n_abacus) * 720
Z = np.random.random(n_abacus) * 720

MST = mist.GetMST(x = X, y = Y, z = Z)
MST_histogram = mist.HistMST()
MST_histogram.setup(usenorm = False, uselog = True)
d, l, b, s, l_index, b_index = MST.get_stats(include_index=True)
MST_histogram = MST_histogram.get_hist(d, l, b, s)

density = MST_histogram['x_d'] / (720**3)
print(density)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Full_Random"

np.savetxt(str(target) + str(i) + "_X_d", MST_histogram['x_d'])
np.savetxt(str(target) + str(i) + "_Y_d", MST_histogram['y_d'])
np.savetxt(str(target) + str(i) + "_X_l", MST_histogram['x_l'])
np.savetxt(str(target) + str(i) + "_Y_l", MST_histogram['y_l'])
np.savetxt(str(target) + str(i) + "_X_b", MST_histogram['x_b'])
np.savetxt(str(target) + str(i) + "_Y_b", MST_histogram['y_b'])
np.savetxt(str(target) + str(i) + "_X_s", MST_histogram['x_s'])
np.savetxt(str(target) + str(i) + "_Y_s", MST_histogram['y_s'])

# then with systematics

X_masked, Y_masked, Z_masked = [], [], []
for i in range(n_abacus):
    if (((X[i] - 360)**2 + (Y[i] - 360)**2 + (Z[i] - 360)**2) < 1600):
        X_masked.append(X[i])
        Y_masked.append(Y[i])
        Z_masked.append(Z[i])
X_masked = np.array(X_masked)
Y_masked = np.array(Y_masked)
Z_masked = np.array(Z_masked)

MST = mist.GetMST(x = X_masked, y = Y_masked, z = Z_masked)
MST_histogram = mist.HistMST()
MST_histogram.setup(usenorm = False, uselog = True)
d, l, b, s, l_index, b_index = MST.get_stats(include_index=True)
MST_histogram = MST_histogram.get_hist(d, l, b, s)

density = MST_histogram['x_d'] / (720**3)
print(density)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Masked_Random"

np.savetxt(str(target) + str(i) + "_X_d", MST_histogram['x_d'])
np.savetxt(str(target) + str(i) + "_Y_d", MST_histogram['y_d'])
np.savetxt(str(target) + str(i) + "_X_l", MST_histogram['x_l'])
np.savetxt(str(target) + str(i) + "_Y_l", MST_histogram['y_l'])
np.savetxt(str(target) + str(i) + "_X_b", MST_histogram['x_b'])
np.savetxt(str(target) + str(i) + "_Y_b", MST_histogram['y_b'])
np.savetxt(str(target) + str(i) + "_X_s", MST_histogram['x_s'])
np.savetxt(str(target) + str(i) + "_Y_s", MST_histogram['y_s'])