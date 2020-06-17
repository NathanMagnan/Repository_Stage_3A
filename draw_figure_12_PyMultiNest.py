## Imports
import numpy as np
import math as m
import GPy as GPy
import pymultinest as pmn
import json as json
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
#sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools as GP
os.chdir('/home/astro/magnan')
#os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## plotting the results
print("Starting to plot the results")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/MCMC')
#my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/MCMC')
target = 'Abacus_ms_2_'
target = os.path.join(my_path, target)

os.system('python3' + ' ' + '/home/astro/magnan/PyMultiNest/multinest_marginals_fancy.py' + ' ' + target)

print("Results plotted")