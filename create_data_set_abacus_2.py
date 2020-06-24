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

## creating the data set
print("starting to work on creating the data set")

for i in range(30, 40):
	print("starting to work on simulation " + str(i))
    
	if (i != 41):
		# getting the basepath
		if (i < 10):
				number_str = str(0) + str(i)
		elif (i<40):
				number_str = str(i)
		elif (i == 40):
				number_str = 'planck'
		path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
		path += number_str
		path += '_products/AbacusCosmos_720box_'
		path += number_str
		path += '_rockstar_halos/z0.100'
        
		# reading the simulation parameters
		input_path = path + '/header'
		param = InputFile.InputFile(fn = input_path)
        
		h0 = (param.get('H0') - 60)  / (75 - 60) # we normalize every parameter to use the whole range from 0 to 1
		w0 = (param.get('w0') - (-1.40)) / ((-0.60) - (-1.40))
		ns = (param.get('ns') - 0.920) / (0.995 - 0.920)
		sigma8 = (param.get('sigma_8') - 0.64) / (1.04 - 0.64)
		omegaM = (param.get('Omega_M') - 0.250) / (0.375 - 0.250)
        
		# getting the reduced MST statistics
		ab = cat.Catalogue_Abacus(basePath = path)
		print("simulation " + str(i) + " : catalogue created")
        
		ab.initialise_data()
		print("simulation " + str(i) + " : data acquired")
        
		current_density = np.shape(ab.CM)[0] / (720**3)
		objective_density = 0.00039769792
		n_galaxies_to_keep = int(objective_density * (720**3))
        
		Indexes_to_keep = ab.Masses.argsort()[-n_galaxies_to_keep:]
		New_CM = []
		for index in Indexes_to_keep:
				New_CM.append(ab.CM[index])
		New_CM = np.asarray(New_CM)
        
		ab.CM = New_CM
		print("simulation " + str(i) + " : catalogue reduced to the BigMD density")
        
		ab.compute_MST_histogram(jacknife = False)
		print("simulation " + str(i) + " : histogram computed")
        
		# extracting the MST statistics and renormalizing them
		X_d = ab.MST_histogram['x_d']
		Y_d = ab.MST_histogram['y_d'] / np.max(ab.MST_histogram['y_d']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_l = ab.MST_histogram['x_l']
		Y_l = ab.MST_histogram['y_l'] / np.max(ab.MST_histogram['y_l']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_b = ab.MST_histogram['x_b']
		Y_b = ab.MST_histogram['y_b'] / np.max(ab.MST_histogram['y_b']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_s = ab.MST_histogram['x_s']
		Y_s = ab.MST_histogram['y_s'] / np.max(ab.MST_histogram['y_s']) # we normalize to 1 to avoid probleme with different box sizes
        
	else :
		# getting the basepath
		my_path = '/hpcstorage/zhaoc/BOSS_PATCHY_MOCKS/ref/'
		my_file = 'Box_HAM_z0.465600_nbar3.976980e-04_scat0.2384.dat'
		my_file = os.path.join(my_path, my_file)
        
		# getting the simulation parameters
		h0 = (67.77 - 60)  / (75 - 60)
		w0 = 0.500 # Not known for BigMD
		ns = (0.9611 - 0.920) / (0.995 - 0.920)
		sigma8 = (0.8288 - 0.64) / (1.04 - 0.64)
		omegaM = (0.307115 + 0.048206 - 0.250) / (0.375 - 0.250)
        
		# getting the galaxies
		BigMD = pandas.read_csv(my_file, sep = ' ', names = ['X', '0', 'Y', '1', 'Z', '2', 'vX', '3', 'vY', '4', 'vZ', '5', 'vMax', '6', 'Other'])
		X = BigMD['X'].values
		Y = BigMD['Y'].values
		Z = BigMD['Z'].values
        
		# getting the MST statistics
		BigMD_mst = mist.GetMST(x = X, y = Y, z = Z)
		d, l, b, s = BigMD_mst.get_stats(include_index = False)
		BigMD_histogram = mist.HistMST()
		BigMD_histogram.setup(usenorm = False, uselog = True)
		BigMD_histogram = BigMD_histogram.get_hist(d, l, b, s)
        
		# extracting the MST statistics and renormalizing them
		X_d = BigMD_histogram['x_d']
		Y_d = BigMD_histogram['y_d'] / np.max(BigMD_histogram['y_d']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_l = BigMD_histogram['x_l']
		Y_l = BigMD_histogram['y_l'] / np.max(BigMD_histogram['y_l']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_b = BigMD_histogram['x_b']
		Y_b = BigMD_histogram['y_b'] / np.max(BigMD_histogram['y_b']) # we normalize to 1 to avoid probleme with different box sizes
        
		X_s = BigMD_histogram['x_s']
		Y_s = BigMD_histogram['y_s'] / np.max(BigMD_histogram['y_s']) # we normalize to 1 to avoid probleme with different box sizes
    
	# saving the full statistics
	target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_2/MST_stats_Catalogue_"
	#target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus_2/MST_stats_Catalogue_"
    
	np.savetxt(str(target) + str(i) + "_X_d", X_d)
	np.savetxt(str(target) + str(i) + "_Y_d", Y_d)
    
	np.savetxt(str(target) + str(i) + "_X_l", X_l)
	np.savetxt(str(target) + str(i) + "_Y_l", Y_l)
    
	np.savetxt(str(target) + str(i) + "_X_b", X_b)
	np.savetxt(str(target) + str(i) + "_Y_b", Y_b)
    
	np.savetxt(str(target) + str(i) + "_X_s", X_s)
	np.savetxt(str(target) + str(i) + "_Y_s", Y_s)
    
	# selecting the points
	X_d_kept = X_d[0:4] / 6 # we normalize X axis between 0 and 1, and we keep only points where BigMD and Abacus agree and where there is sensitivity to cosmology
	Y_d_kept = np.log10(Y_d[0:4]) # we normalize Y axis to help the GP
    
	X_l_kept = [] # we keep only points where BigMD and Abacus agree and where there is sensitivity to cosmology
	Y_l_kept = []
	for j in range(len(X_l)):
		if ((X_l[j] > 3) and (X_l[j] < 10)):
				X_l_kept.append(X_l[j])
				Y_l_kept.append(Y_l[j])
	X_l_kept = np.asarray(X_l_kept)
	Y_l_kept = np.asarray(Y_l_kept)
	X_l_kept = X_l_kept[::2] # we try to avoid too much redundant information
	Y_l_kept = Y_l_kept[::2]
	X_l_kept = X_l_kept[:12] # we make sure every l histogram has the same size
	Y_l_kept = Y_l_kept[:12]
	X_l_kept = np.log10(X_l_kept) # we normalize X axis between 0 and 1
	Y_l_kept = np.log10(Y_l_kept) # we normalize Y axis to help the GP
    
	X_b_kept = [] # we keep only points where BigMD and Abacus agree and where there is sensitivity to cosmology
	Y_b_kept = []
	for j in range(len(X_b)):
		if ((X_b[j] > 10) and (X_b[j] < 50)):
				X_b_kept.append(X_b[j])
				Y_b_kept.append(Y_b[j])
	X_b_kept = np.asarray(X_b_kept)
	Y_b_kept = np.asarray(Y_b_kept)
	X_b_kept = X_b_kept[::3] # we try to avoid too much redundant information
	Y_b_kept = Y_b_kept[::3]
	X_b_kept = X_b_kept[:11] # we make sure every l histogram has the same size
	Y_b_kept = Y_b_kept[:11]
	X_b_kept = np.log10(X_b_kept) # we normalize X axis between 0 and 1
	Y_b_kept = np.log10(Y_b_kept) # we normalize Y axis to help the GP
    
	X_s_kept = [] # we keep only points where BigMD and Abacus agree and where there is sensitivity to cosmology
	Y_s_kept = []
	for j in range(len(X_s)):
		if ((X_s[j] > 0.4) and (X_s[j] < 0.7)):
				X_s_kept.append(X_s[j])
				Y_s_kept.append(Y_s[j])
	X_s_kept = np.asarray(X_s_kept)
	Y_s_kept = np.asarray(Y_s_kept)
	X_s_kept = X_s_kept[::3] # we try to avoid too much redundant information
	Y_s_kept = Y_s_kept[::3]
	Y_s_kept = np.log10(Y_s_kept) # we normalize Y axis to help the GP
    
	# creating the set of new points
	X_data_new_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d, 0] for x_d in X_d_kept], (4, 7))
	X_data_new_l = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l, 1] for x_l in X_l_kept], (12, 7))
	X_data_new_b = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b, 2] for x_b in X_b_kept], (11, 7))
	X_data_new_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s, 3] for x_s in X_s_kept], (5, 7))
	X_data_new = np.concatenate((X_data_new_d, X_data_new_l, X_data_new_b, X_data_new_s), 0)
    
	Y_data_new_d = np.reshape([[y_d] for y_d in Y_d_kept], (4, 1))
	Y_data_new_l = np.reshape([[y_l] for y_l in Y_l_kept], (12, 1))
	Y_data_new_b = np.reshape([[y_b] for y_b in Y_b_kept], (11, 1))
	Y_data_new_s = np.reshape([[y_s] for y_s in Y_s_kept], (5, 1))
	Y_data_new = np.concatenate((Y_data_new_d, Y_data_new_l, Y_data_new_b, Y_data_new_s), 0)
    
	# saving the new points
	target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus_2/data_set_Abacus"
    
	np.savetxt(str(target) + "_" + str(i) + "_X_data", X_data_new)
	np.savetxt(str(target) + "_" + str(i) + "_Y_data", Y_data_new)

print("data set fully created")