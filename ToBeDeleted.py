## Imports
import numpy as np

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## creating the data set
print("starting to work on creating the data set")
X_data = []
Y_data = []
Catalogues = []

for i in range(1):
    print("starting to work on simulation " + str(i))
    
    # getting the basepath
    if (i < 10):
        number_str = str(0) + str(i)
    elif (i<40):
        number_str = str(i)
    else:
        number_str = 'planck'
    path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
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
    
    # computing the histogram
    ab.compute_MST_histogram(jacknife = True)
    print("simulation " + str(i) + " : histogram computed")
    
    # saving the catalogue
    Catalogues.append(ab)
    
    # reading the histogram
    print(ab.MST_histogram)