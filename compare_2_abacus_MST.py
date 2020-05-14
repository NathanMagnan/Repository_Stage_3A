import numpy as np

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
os.chdir('/home/astro/magnan')

## getting the 2 abacus sims data
ab00 = cat.Catalogue_Abacus(basePath = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_00_products/AbacusCosmos_720box_00_rockstar_halos/z0.100')
ab39 = cat.Catalogue_Abacus(basePath = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_39_products/AbacusCosmos_720box_39_rockstar_halos/z0.100')
ab00.initialise_data()
ab39.initialise_data()
print("Data loaded")

## getting the MST data
ab00.compute_MST_histogram()
ab39.compute_MST_histogram()
print("histograms computed")

## plotting the comparison
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Comparison_Between_2_Abacus_simulation.png'
my_file = os.path.join(my_path, my_file)
cat.compare_MST_histograms(List_catalogues = [ab00, ab39], usemean = False, whichcomp = 0, title = "Comparison between 2 Abacus simulation's MSTs", saveas = my_file)
print("histograms plotted and saved")
