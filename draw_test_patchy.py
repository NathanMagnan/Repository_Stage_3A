## Imports
import numpy as np
import pickle
import os
import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Loading Patchy
print("starting to load patchy")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_patchy/')
my_file = 'Test_' + '.pkl'
my_file = os.path.join(my_path, my_file)
	
f = open(my_file, "rb")
patchy_histogram = pickle.load(f)
f.close()

print("patchy loaded")

## Loading Abacus (single)

print("Starting to load Abacus (single)")

# getting the basepath
number_str = 'planck'
path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)

# gettting the data
ab.initialise_data()

# reduce the catalogue
current_density = np.shape(ab.CM)[0] / (720**3)
print(current_density)
objective_density = 0.00039769792
print(objective_density)
n_galaxies_to_keep = int(objective_density * (720**3))

Indexes_to_keep = ab.Masses.argsort()[-n_galaxies_to_keep:]
New_CM = []
for index in Indexes_to_keep:
	New_CM.append(ab.CM[index])
New_CM = np.asarray(New_CM)

ab.CM = New_CM
new_density = np.shape(ab.CM)[0] / (720**3)
print(new_density)

# getting the MST stats
ab.compute_MST_histogram(jacknife = False)

print("Abacus loaded")

## Loading Abacus (whole)
print("starting to load Abacus")

target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
#target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
	X_d = np.loadtxt(str(target) + str(i) + "_X_d")
	Y_d = np.loadtxt(str(target) + str(i) + "_Y_d")
	dict['X_d'].append(X_d)
	dict['Y_d'].append(Y_d)
    
	X_l = np.loadtxt(str(target) + str(i) + "_X_l")
	Y_l = np.loadtxt(str(target) + str(i) + "_Y_l")
	dict['X_l'].append(X_l)
	dict['Y_l'].append(Y_l)
    
	X_b = np.loadtxt(str(target) + str(i) + "_X_b")
	Y_b = np.loadtxt(str(target) + str(i) + "_Y_b")
	dict['X_b'].append(X_b)
	dict['Y_b'].append(Y_b)
    
	X_s = np.loadtxt(str(target) + str(i) + "_X_s")
	Y_s = np.loadtxt(str(target) + str(i) + "_Y_s")
	dict['X_s'].append(X_s)
	dict['Y_s'].append(Y_s)

print("Abacus loaded")

## Plot
print("Starting to plot the MST stats")

fig, axes = plt.subplots(ncols = 4, figsize = (20, 8))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
	subplot = axes[j]
	
	if (j == 0):
		subplot.set_xlabel('$d$')
		
		subplot.set_ylabel('$N_{d}$')
		subplot.set_yscale('log')
       
		Mean = dict['Y_d'][0]
		Std = np.asarray([0 for k in range(np.shape(dict['Y_d'][0])[0])])
		for k in range(1, 41):
			Mean_old = Mean.copy()
			Std_old = Std.copy()
           
			Mean_new = (k * Mean_old + dict['Y_d'][k]) / (k + 1)
			Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_d'][k]**2) / (k + 1) - Mean_new**2)
           
			Mean = Mean_new.copy()
			Std = Std_new.copy()
		m1 = np.max(Mean)
		m2 = np.max(patchy_histogram['y_d'])
		m3 = np.max(ab.MST_histogram['y_d'])
       
		subplot.fill_between(x = dict['X_d'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'b', alpha = 0.2)
		subplot.plot(dict['X_d'][0], Mean, 'b', label = "Abacus")
		subplot.plot(patchy_histogram['x_d'], patchy_histogram['y_d'] * m1 / m2, 'k', label = "Patchy")
		subplot.plot(ab.MST_histogram['x_d'], ab.MST_histogram['y_d'] * m1 / m3, 'k--', label = "Abacus reduced")
		subplot.legend()
  
	elif (j == 1):
		subplot.set_xlabel('$l$')
		subplot.set_xscale('log')
		
		subplot.set_ylabel('$N_{l}$')
		subplot.set_yscale('log')
       
		Mean = dict['Y_l'][0]
		Std = np.asarray([0 for k in range(np.shape(dict['Y_l'][0])[0])])
		for k in range(1, 41):
			Mean_old = Mean.copy()
			Std_old = Std.copy()
           
			Mean_new = (k * Mean_old + dict['Y_l'][k]) / (k + 1)
			Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_l'][k]**2) / (k + 1) - Mean_new**2)
           
			Mean = Mean_new.copy()
			Std = Std_new.copy()
		m1 = np.max(Mean)
		m2 = np.max(patchy_histogram['y_l'])
		m3 = np.max(ab.MST_histogram['y_l'])
       
		subplot.fill_between(x = dict['X_l'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'g', alpha = 0.2)
		subplot.plot(dict['X_l'][0], Mean, 'g', label = "Abacus")
		subplot.plot(patchy_histogram['x_l'], patchy_histogram['y_l'] * m1 / m2, 'k', label = "Patchy")
		subplot.plot(ab.MST_histogram['x_l'], ab.MST_histogram['y_l'] * m1 / m3, 'k--', label = "Abacus reduced")
		subplot.legend()
	    
	elif (j == 2):
		subplot.set_xlabel('$b$')
		subplot.set_xscale('log')
		
		subplot.set_ylabel('$N_{b}$')
		subplot.set_yscale('log')
       
		Mean = dict['Y_b'][0]
		Std = np.asarray([0 for k in range(np.shape(dict['Y_b'][0])[0])])
		for k in range(1, 41):
			Mean_old = Mean.copy()
			Std_old = Std.copy()
           
			Mean_new = (k * Mean_old + dict['Y_b'][k]) / (k + 1)
			Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_b'][k]**2) / (k + 1) - Mean_new**2)
           
			Mean = Mean_new.copy()
			Std = Std_new.copy()
		m1 = np.max(Mean)
		m2 = np.max(patchy_histogram['y_b'])
		m3 = np.max(ab.MST_histogram['y_b'])
       
		subplot.fill_between(x = dict['X_b'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'r', alpha = 0.2)
		subplot.plot(dict['X_b'][0], Mean, 'r', label = "Abacus")
		subplot.plot(patchy_histogram['x_b'], patchy_histogram['y_b'] * m1 / m2, 'k', label = "Patchy")
		subplot.plot(ab.MST_histogram['x_b'], ab.MST_histogram['y_b'] * m1 / m3, 'k--', label = "Abacus reduced")
		subplot.legend()
	          
	else:
		subplot.set_xlabel('$s$')
		
		subplot.set_ylabel('$N_{s}$')
		subplot.set_yscale('log')
       
		Mean = dict['Y_s'][0]
		Std = np.asarray([0 for k in range(np.shape(dict['Y_s'][0])[0])])
		for k in range(1, 41):
			Mean_old = Mean.copy()
			Std_old = Std.copy()
           
			Mean_new = (k * Mean_old + dict['Y_s'][k]) / (k + 1)
			Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_s'][k]**2) / (k + 1) - Mean_new**2)
           
			Mean = Mean_new.copy()
			Std = Std_new.copy()
		m1 = np.max(Mean)
		m2 = np.max(patchy_histogram['y_s'])
		m3 = np.max(ab.MST_histogram['y_s'])
       
		subplot.fill_between(x = dict['X_s'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'y', alpha = 0.2)
		subplot.plot(dict['X_s'][0], Mean, 'y', label = "Abacus")
		subplot.plot(patchy_histogram['x_s'], patchy_histogram['y_s'] * m1 / m2, 'k', label = "Patchy")
		subplot.plot(ab.MST_histogram['x_s'], ab.MST_histogram['y_s'] * m1 / m3, 'k--', label = "Abacus reduced")
		subplot.legend()

plt.suptitle("Comparison of Abacus MSTS's to a patchy mock")
print("results plotted")

print("starting to save the results")
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
#my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Test_patchy_mocks.png'
plt.savefig(os.path.join(my_path, my_file))
print("results saved")

plt.show()