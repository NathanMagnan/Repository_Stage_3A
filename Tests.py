import numpy as np
import GPy as GPy

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
import GP_tools as GP
os.chdir('/home/astro/magnan')
# imports OK

""" Testing Catalogue Tools """
#alf = cat.Catalogue_ALF(count = 10**4, alpha = 1.5, beta = 0.4, gamma = 1.3, t0 = 0.3, ts = 0.01, box_size = 75.)
#ab = cat.Catalogue_Abacus()
#il = cat.Catalogue_Illustris()
# __init__ OK

#alf.initialise_data()
#ab.initialise_data()
#il.initialise_data()
# initialise_data OK but quite slow (likely because it needs to load simulations)

#alf.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#ab.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#il.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
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

#alf.plot_2PCF(title = "ALF reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#ab.plot_2PCF(title = "Abacus reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#il.plot_2PCF(title = "Illustris reliable 2PCF", full_output = False, min_reliable = 0.1, max_reliable = 200, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#alf.plot_2PCF(title = "ALF 2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#ab.plot_2PCF(title = "Abacus  2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#il.plot_2PCF(title = "Illustris 2PCF", full_output = True, bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
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

#d, b, l, s, l_index, b_index = alf.MST.get_stats(include_index = True)
#print(l_index)

#alf.compute_MST_histogram(mode_MST = 'SingleMST')
#alf.plot_MST_histogram(title = "Single MST histogram")
#alf.compute_MST_histogram(mode_MST = 'MultipleMST')
#alf.plot_MST_histogram(title = "Statistical MST histogram")
#ab.plot_MST_histogram(title = "Single MST histogram")
#il.plot_MST_histogram(title = "Single MST histogram")
# plot_MST_histogram works partially : no title AND very slow for Abacus (more than 1 min)

#alf.plot_MST_2D(title = "2D plot of ALF MST")
#ab.plot_MST_2D(title = "2D plot of Abacus MST")
#il.plot_MST_2D(title = "2D plot of Illustris MST")
# plot_MST_2D OK BUT there seem to be an issue with the MST calculation : one can create a ALF with non-connected MST and even some points without any edge...

#alf.plot_MST_3D(title = "3D plot of ALF MST")
#ab.plot_MST_3D(title = "3D plot of Abacus MST")
#il.plot_MST_3D(title = "3D plot of Illustris MST")
# plot_MST_3D seems to be OK but really slow

#alf.compute_HMF(bin_min_HMF = 10**9, bin_max_HMF = 10**11, n_bin_HMF = 100)
#ab.compute_HMF(bin_min_HMF = 10**9, bin_max_HMF = 10**15, n_bin_HMF = 100)
#il.compute_HMF(bin_min_HMF = 10**7, bin_max_HMF = 10**15, n_bin_HMF = 100)
# compute_HMF seems OK but there a small warning for alf

#alf.plot_HMF(title = "Halo Mass Function for ALF")
#ab.plot_HMF(title = "Halo Mass Function for Abacus")
#il.plot_HMF(title = "Halo Mass Function for Illustris")
# plot_HMF OK

#alf.plot_HMF(title = "Halo Mass Function for ALF", bin_min_HMF = 10**9, bin_max_HMF = 10**11, n_bin_HMF = 100)
#ab.plot_HMF(title = "Halo Mass Function for Abacus", bin_min_HMF = 10**9, bin_max_HMF = 10**17, n_bin_HMF = 100)
#il.plot_HMF(title = "Halo Mass Function for Illustris", bin_min_HMF = 10**11, bin_max_HMF = 10**15, n_bin_HMF = 100)
# plot_HMF OK in pipelined mode

#alf.compute_HMF(bin_min_HMF = 10**9, bin_max_HMF = 10**11, n_bin_HMF = 100)
#ab.compute_HMF(bin_min_HMF = 10**9, bin_max_HMF = 10**17, n_bin_HMF = 100)
#il.compute_HMF(bin_min_HMF = 10**7, bin_max_HMF = 10**15, n_bin_HMF = 100)
#cat.compare_HMFs(List_catalogues = [alf, ab, il], title = "Comparison between each catalogue Halo Mass Function")
# compare_HMFs OK

#alf.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#ab.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#il.compute_2PCF(bin_min = 0.01, bin_max = 1000, n_bin_2PCF = 100)
#cat.compare_2PCFs(List_catalogues = [alf, ab, il], title = "Comparison between each catalogue 2PCF")
# compare_2PCFs OK

#alf.compute_MST_histogram(mode_MST = 'MultipleMST')
#ab.compute_MST_histogram()
#il.compute_MST_histogram()
#cat.compare_MST_histograms(List_catalogues = [alf, ab, il], title = "Comparison between each catalogue MSTs")
# compare_MST_histograms OK BUT no title and if different counts the results are unreadable...

""" Testing GP Tools """

#print("starting to work on creating the data set")
#X_data = []
#Y_data = []
#
#for i in range(40):
#    print("starting to work on simulation " + str(i))
#    
#    # creating a catalogue object
#    alf = cat.Catalogue_ALF(count = 10**4, alpha = 1.5, beta = 0.4, gamma = 1.3, t0 = 0.3, ts = 0.01, box_size = 75.)
#    print("simulation " + str(i) + " : catalogue created")
#    
#    # gettting the data
#    alf.initialise_data()
#    print("simulation " + str(i) + " : data acquired")
#    
#    # computing the histogram
#    alf.compute_MST_histogram()
#    print("simulation " + str(i) + " : histogram computed")
#    
#    # reading the histogram
#    X_d = alf.MST_histogram['x_d']
#    Y_d = alf.MST_histogram['y_d'] / np.max(alf.MST_histogram['y_d']) # we normalize every histogram to 1
#    
#    X_l_long = alf.MST_histogram['x_l']
#    Y_l_long = alf.MST_histogram['y_l'] / np.max(alf.MST_histogram['y_l']) # we normalize every histogram to 1
#    X_l = X_l_long[4::10]
#    Y_l = Y_l_long[4::10] # we only keep 10 points per histogram to avoid having huge redundant dataset
#    
#    X_b_long = alf.MST_histogram['x_b']
#    Y_b_long = alf.MST_histogram['y_b'] / np.max(alf.MST_histogram['y_b']) # we normalize every histogram to 1
#    X_b = X_b_long[4::10]
#    Y_b = Y_b_long[4::10] # we only keep 10 points per histogram to avoid having huge redundant dataset
#    
#    X_s_long = alf.MST_histogram['x_s']
#    Y_s_long = alf.MST_histogram['y_s'] / np.max(alf.MST_histogram['y_s']) # we normalize every histogram to 1
#    X_s = X_s_long[2::5]
#    Y_s = Y_s_long[2::5] # we only keep 10 points per histogram to avoid having huge redundant dataset
#    print("simulation " + str(i) + " : histogram read")
#    
#    # reading the simulation parameters
#    
#    h0 = np.random.random()
#    w0 = np.random.random()
#    ns = np.random.random()
#    sigma8 = np.random.random()
#    omegaM = np.random.random()
#    print("simulation " + str(i) + " : parameters read")
#    
#    # creating the set of new points
#    X_data_new_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d, 0] for x_d in X_d], (6, 7))
#    X_data_new_l = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l, 1] for x_l in X_l], (10, 7))
#    X_data_new_b = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b, 2] for x_b in X_b], (10, 7))
#    X_data_new_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s, 3] for x_s in X_s], (10, 7))
#    X_data_new = np.concatenate((X_data_new_d, X_data_new_l, X_data_new_b, X_data_new_s), 0)
#    
#    Y_data_new_d = np.reshape([[y_d] for y_d in Y_d], (6, 1))
#    Y_data_new_l = np.reshape([[y_l] for y_l in Y_l], (10, 1))
#    Y_data_new_b = np.reshape([[y_b] for y_b in Y_b], (10, 1))
#    Y_data_new_s = np.reshape([[y_s] for y_s in Y_s], (10, 1))
#    Y_data_new = np.concatenate((Y_data_new_d, Y_data_new_l, Y_data_new_b, Y_data_new_s), 0)
#    print("simulation " + str(i) + " : new data points created")
#    
#    # adding the new points
#    if (i == 0):
#        X_data = X_data_new
#        Y_data = Y_data_new
#    else:
#        X_data = np.concatenate((X_data, X_data_new), 0)
#        Y_data = np.concatenate((Y_data, Y_data_new), 0)
#    print("simulation " + str(i) + " : new data points added to the data set")
#
#print("data set fully created")
#
### saving the data set
#print("starting to work on saving the data set")
#
#target = "/home/astro/magnan/Repository_Stage_3A/data_set_test"
#
#np.savetxt(str(target) + "_X_data", X_data)
#np.savetxt(str(target) + "_Y_data", Y_data)
#
#print("data fully saved")
#print("Data set ready !")

#print("starting to load the data")
#
#target = "/home/astro/magnan/Repository_Stage_3A/data_set_test"
#
#X_data = np.loadtxt(str(target) + "_X_data")
#Y_data = np.loadtxt(str(target) + "_Y_data")
#
#print("data loaded")
#
#kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5]) # the column 6 will be dealt with by the coregionalization
#kernel_output = GPy.kern.Coregionalize(input_dim = 1, output_dim = 4, rank = 4) # rank 4 since there are 4 outputs
#kernel = kernel_input**kernel_output
#print("kernel sucessfully constructed")
#print(kernel)
#
#n_test = 4
#n_simulations = 40
#n_points_per_simulation = 36
#
#start_group = (n_simulations - n_test) * n_points_per_simulation
#end_group =  n_simulations * n_points_per_simulation
#
#X_test_group = X_data[start_group:end_group]
#X_data_group = np.concatenate((X_data[0:start_group], X_data[end_group:]), 0)
#Y_test_group = Y_data[start_group:end_group]
#Y_data_group = np.concatenate((Y_data[0:start_group], Y_data[end_group:]), 0)
#
## Noise_data = np.identity(n = np.shape(X_data_group)[0]) * 0.05
## Noise_test = np.identity(n = np.shape(X_test_group)[0]) * 0.05
#noise_data = 0.05
#noise_test = 0.05
#print("Data and test group successfully created")
#
## gp = GP.GP(X_data_group, Y_data_group, kernel = kernel, Noise_data = Noise_data)
#gp = GP.GP(X_data_group, Y_data_group, kernel = kernel, noise_data = noise_data)
#print("GP successfully created")
#
#gp.initialise_model()
#print("GP model successfully created")
#print(gp.model)
#
#gp.optimize_model()
#print("GP model successfully optimized")
#print(gp.model)
#
## performance = gp.compute_performance_on_tests(X_test = X_test_group, Y_test = Y_test_group, Noise_test = Noise_test)
#performance = gp.compute_performance_on_tests(X_test = X_test_group, Y_test = Y_test_group, noise_test = noise_test)
#print("performance successfully computed")
#print(performance)
#
#prediction, Cov = gp.compute_prediction([[0.7, 0.3, 0.5, 0.4, 0.1]])
#print("prediction successfully computed")
#print(prediction)
#print(Cov)

