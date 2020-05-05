import numpy as np
import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
os.chdir('/home/astro/magnan')
# imports OK

alf = cat.Catalogue_ALF(count = 10**4, alpha = 1.5, beta = 0.4, gamma = 1.3, t0 = 0.3, ts = 0.01, box_size = 75.)
ab = cat.Catalogue_Abacus()
il = cat.Catalogue_Illustris()
# __init__ OK

alf.initialise_data()
ab.initialise_data()
il.initialise_data()
# initialise_data OK but really slow

#alf.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin = 100)
#ab.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin = 100)
#il.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin = 100)
# compute_2PCF OK

#alf.extract_reliable_2PCF(min_reliable = 0.1, max_reliable = 200)
#ab.extract_reliable_2PCF(min_reliable = 0.1, max_reliable = 200)
#il.extract_reliable_2PCF(min_reliable = 0.1, max_reliable = 200)
# extract_reliable_2PCF OK

#print("Trying to plot ALF 2PCF")
#alf.plot_2PCF(title = "ALF entire 2PCF", full_output = True)
#print("Trying to plot Abacus 2PCF")
#ab.plot_2PCF(title = "Abacus entire 2PCF", full_output = True)
#print("Trying to plot Illustris 2PCF")
#il.plot_2PCF(title = "Illustris entire 2PCF", full_output = True)
# plot_2PCF(full_output = True) OK

#print("Trying to plot ALF 2PCF")
#alf.plot_2PCF(title = "ALF reliable 2PCF", full_output = False)
#print("Trying to plot Abacus 2PCF")
#ab.plot_2PCF(title = "Abacus reliable 2PCF", full_output = False)
#print("Trying to plot Illustris 2PCF")
#il.plot_2PCF(title = "Illustris reliable 2PCF", full_output = False)
# plot_2PCF(full_output = False) OK

#alf.plot_2PCF(title = "ALF reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin = 100)
#ab.plot_2PCF(title = "Abacus reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin = 100)
#il.plot_2PCF(title = "Illustris reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin = 100)
#alf.plot_2PCF(title = "ALF reliable 2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin = 100)
#ab.plot_2PCF(title = "Abacus reliable 2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin = 100)
#il.plot_2PCF(title = "Illustris reliable 2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin = 100)
# pipelined version of plot_2PCF OK

#alf.plot_2D(title = "2D plot for ALF")
#ab.plot_2D(title = "2D plot for Abacus")
#il.plot_2D(title = "2D plot for Illustris")
# plot_2D OK in both pipeline and standard mode

#alf.plot_3D(title = "3D plot for ALF")
#ab.plot_3D(title = "3D plot for Abacus")
#il.plot_3D(title = "3D plot for Illustris")
# plot_3D OK in both pipeline and standard mode

#alf.compute_MST()
#ab.compute_MST()
#il.compute_MST()
# compute_MST OK

#alf.compute_MST_histogram(mode_MST = 'SingleMST')
#alf.compute_MST_histogram(mode_MST = 'MultipleMST')
#ab.compute_MST_histogram()
#il.compute_MST_histogram()
# compute_MST_histrogram OK

#alf.compute_MST_histogram(mode_MST = 'SingleMST')
#alf.plot_MST_histogram(title = "Single MST histogram")
#alf.compute_MST_histogram(mode_MST = 'MultipleMST')
#alf.plot_MST_histogram(title = "Statistical MST histogram")
#ab.plot_MST_histogram(title = "Single MST histogram")
#il.plot_MST_histogram(title = "Single MST histogram")
# compute_MST_histogram works partially : no title

#alf.plot_MST_2D(title = "2D plot of ALF MST")
#ab.plot_MST_2D(title = "2D plot of Abacus MST")
#il.plot_MST_2D(title = "2D plot of Illustris MST")
# plot_MST_2D OK BUT there seem to be an issue with the MST calculation : one can create a ALF with non-connected MST and even some points without any edge...

#alf.plot_MST_3D(title = "3D plot of ALF MST")
#ab.plot_MST_3D(title = "3D plot of Abacus MST")
#il.plot_MST_3D(title = "3D plot of Illustris MST")
# plot_MST_3D seems to be OK but really slow

