## Imports
import numpy as np
import mistree as mist

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import illustris_tools as il
import abacus_tools as ab
import alf_tools as alf
os.chdir('/home/astro/magnan')

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)
from mpl_toolkits.mplot3d import Axes3D

## Catalogue Class
class Catalogue():
    
    def __init__(self):
        self.type = None # string -  Abacus, Illustris, ALF
        self.color = None # string - blue for Abacus, green for Illustris, red for ALF
        self.count = None # float -  number of galaxies
        self.box_size = None # float - Mpc / h
        self.CM = None # np.array(count, 3) - Mpc / h
        self.h = None # float - km/s / Mpc
        self.Masses # float - M_sun / h
        
        self.parameters_HMF = None #dict(bin_min_HMF, bin_max_HMF, n_bin_HMF) - first two in M_sun / h
        self.Bins_HMF = None # np.array(n_bin_HMF, 1)
        self.HMF = None # np.array(n_bin_HMF, 1)
        
        self.parameters_2PCF = None # dict(bin_min, bin_max, n_bin_2PCF, min_reliable, max_reliable) - first two and last two in Mpc / h
        self.Bins = None # np.array(n_bin_2PCF, 1) - Mpc / h
        self.Mean_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Std_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Bins_reliable = None # np.array(? , 1) # Mpc / h
        self.Mean_2PCF_reliable = None # np.array(?, 1)
        self.Std_2PCF_reliable = None # np.array(?, 1)
        
        self.MST = None # mist.mst
        self.MST_histogram = None # dict
    
    def initialise_data(self):
        return("initialise_data To Be Done for " + self.type)
    
    def compute_HMF(self, bin_min_HMF, bin_max_HMF, n_bin_HMF): # does not make much sense for an ALF since it does not include any mass
        if (self.Masses is None):
            self.initialise_data()
        
        self.parameters_HMF = {'bin_min_HMF' : bin_min_HMF, 'bin_max_HMF' : bin_max_HMF, 'n_bin_HMF' : n_bin_HMF}
        self.Bins_HMF = np.logspace(np.log10(bin_min_HMF), np.log10(bin_max_HMF), n_bin_HMF + 1)
        self.HMF, bins = np.histogram(self.Masses, bins = self.Bins_HMF)
    
    def compute_2PCF(self):
        return("compute_2PCF To Be Done for " + self.type)
    
    def extract_reliable_2PCF(self, min_reliable, max_reliable, bin_min = None, bin_max = None, n_bin_2PCF = None):
        if (self.Mean_2PCF is None):
            self.compute_2PCF(bin_min, bin_max, n_bin_2PCF)
        
        self.parameters_2PCF['min_reliable'] = min_reliable
        self.parameters_2PCF['max_reliable'] = max_reliable
        
        Bins_reliable = []
        Mean_2PCF_reliable = []
        Std_2PCF_reliable = []
        for i in range(self.parameters_2PCF['n_bin_2PCF']):
            if ((self.Bins[i] > min_reliable) and (self.Bins[i] < max_reliable)):
                Bins_reliable.append(self.Bins[i])
                Mean_2PCF_reliable.append(self.Mean_2PCF[i])
                Std_2PCF_reliable.append(self.Std_2PCF[i])
        self.Bins_reliable = np.asarray(Bins_reliable)
        self.Mean_2PCF_reliable = np.asarray(Mean_2PCF_reliable)
        self.Std_2PCF_reliable = np.asarray(Std_2PCF_reliable)
    
    def compute_MST(self, jacknife = False):
        if (self.CM is None):
            self.initialise_data()
        
        X = np.asarray(self.CM[:, 0])
        Y = np.asarray(self.CM[:, 1])
        Z = np.asarray(self.CM[:, 2])
        
        self.MST = mist.GetMST(x = X, y = Y, z = Z)
    
    def compute_MST_histogram(self):
        return("compute_MST_histogram To Be Done for " + self.type)
    
    def plot_2D(self, title = " ", figure = None): # slice et Z < 50 Mpc / h
        if (self.CM is None):
            self.initialise_data()
        
        plot_at_the_end = False
        if (figure is None):
        		misleading = plt.figure()
        		figure = misleading.gca()
        		plot_at_the_end = True
        		
        
        figure.set_title(title)
        figure.set_xlabel("$X [h^{-1} Mpc]$")
        figure.set_ylabel("$Y [h^{-1} Mpc]$")
        
        X_slice, Y_slice, Z_slice = [], [], []
        for i in range(self.count):
            if (self.CM[i][2] <= 50):
                X_slice.append(self.CM[i][0])
                Y_slice.append(self.CM[i][1])
                Z_slice.append(self.CM[i][2])
        X_slice = np.asarray(X_slice)
        Y_slice = np.asarray(Y_slice)
        Z_slice = np.asarray(Z_slice)
        figure.scatter(X_slice, Y_slice, c = Z_slice, cmap = 'Greys', s = 0.05)
        
        if plot_at_the_end:
        	   plt.show(block = True)
    
    def plot_3D(self, title = " "): # maximum of 100 000 points
        if (self.CM is None):
            self.initialise_data()
        
        fig = plt.figure()
        
        ax = fig.gca(projection = '3d')
        ax.set_title(title)
        ax.set_xlabel("$X [h^{-1} Mpc]$")
        ax.set_ylabel("$Y [h^{-1} Mpc]$")
        ax.set_zlabel("$Z [h^{-1} Mpc]$")

        List_indexes = np.arange(self.count)
        Indexes_kept = np.random.choice(List_indexes, (100000, ))
        X_reduced, Y_reduced, Z_reduced = [], [], []
        for i in range(100000):
            X_reduced.append(self.CM[Indexes_kept[i]][0])
            Y_reduced.append(self.CM[Indexes_kept[i]][1])
            Z_reduced.append(self.CM[Indexes_kept[i]][2])
        X_reduced = np.asarray(X_reduced)
        Y_reduced = np.asarray(Y_reduced)
        Z_reduced = np.asarray(Z_reduced)
        ax.scatter(X_reduced, Y_reduced, Z_reduced, s = 0.05, color = 'k')
        
        plt.show(block = True)
    
    def plot_HMF(self, title = " ", bin_min_HMF = None, bin_max_HMF = None, n_bin_HMF = None, figure = None):
        if (self.HMF is None):
            self.compute_HMF(bin_min_HMF = bin_min_HMF, bin_max_HMF = bin_max_HMF, n_bin_HMF = n_bin_HMF)
        
        plot_at_the_end = False
        if (figure is None):
        		misleading = plt.figure()
        		figure = misleading.gca()
        		plot_at_the_end = True
        
        figure.set_title(title)
        figure.set_xlabel("Mass [$M_{\odot}$]")
        figure.set_ylabel("Count")
        figure.set_xscale('log')
        figure.set_yscale('log')
        
        figure.plot(self.Bins_HMF[:-1], self.HMF, color = self.color)
        
        if plot_at_the_end:
        	   plt.show(block = True)
        
    
    def plot_2PCF(self, title = " ", full_output = True, bin_min = None, bin_max = None, n_bin_2PCF = None, min_reliable = None, max_reliable = None, figure = None):
        plot_at_the_end = False
        if (figure is None):
        		misleading = plt.figure()
        		figure = misleading.gca()
        		plot_at_the_end = True
        
        figure.set_title(title)
        figure.set_xlabel("$r [h^{-1} Mpc]$")
        figure.set_ylabel("$\\xi(r)$")
        figure.set_xscale('log')
        figure.set_yscale('log')
        
        if (full_output == True):
            if (self.Mean_2PCF is None):
                self.compute_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF)
            figure.plot(self.Bins, self.Mean_2PCF, self.color)
            figure.plot(self.Bins, self.Mean_2PCF - self.Std_2PCF, color = self.color, linestyle = '--')
            figure.plot(self.Bins, self.Mean_2PCF + self.Std_2PCF, color = self.color, linestyle = '--')
        else:
            if (self.Mean_2PCF_reliable is None):
                self.extract_reliable_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF, min_reliable = min_reliable, max_reliable = max_reliable)
            figure.plot(self.Bins_reliable, self.Mean_2PCF_reliable, self.color)
            figure.plot(self.Bins_reliable, self.Mean_2PCF_reliable - self.Std_2PCF_reliable, color = self.color, linestyle = '--')
            figure.plot(self.Bins_reliable, self.Mean_2PCF_reliable + self.Std_2PCF_reliable, color = self.color, linestyle = '--')
        
        if plot_at_the_end:
        		plt.show(block = True)
    
    def plot_MST_2D(self, title = " ", figure = None): # slice at Z < 50 Mpc / h
        if (self.MST is None):
            self.compute_MST()
        
        plot_at_the_end = False
        if (figure is None):
        		misleading = plt.figure()
        		figure = misleading.gca()
        		plot_at_the_end = True
        
        figure.set_title(title)
        figure.set_xlabel('$X [h^{-1} Mpc]$')
        figure.set_ylabel('$Y [h^{-1} Mpc]$')
        
        # plotting nodes:
        X_slice, Y_slice, Z_slice = [], [], []
        for i in range(self.count):
            if (self.CM[i][2] <= 50):
                X_slice.append(self.CM[i][0])
                Y_slice.append(self.CM[i][1])
                Z_slice.append(self.CM[i][2])
        X_slice = np.asarray(X_slice)
        Y_slice = np.asarray(Y_slice)
        Z_slice = np.asarray(Z_slice)
        figure.scatter(X_slice, Y_slice, s=0.2, color='r')
        
        # plotting MST edges:
        d, l, b, s, l_index, b_index = self.MST.get_stats(include_index = True)
        for i in range(np.shape(l_index)[1]): # safer than count - 1 because with the k-nearest neighbors approach the MST might be non - connected
            if ((self.CM[l_index[0][i]][2] <= 50) and (self.CM[l_index[1][i]][2] <= 50)):
                X = [self.CM[l_index[0][i]][0], self.CM[l_index[1][i]][0]]
                Y = [self.CM[l_index[0][i]][1], self.CM[l_index[1][i]][1]]
                figure.plot(X, Y, color = 'k', linewidth = 0.5)
        
        if plot_at_the_end:
        		plt.show(block = True)
    
    def plot_MST_3D(self, title = " "): # maximum of 100 000 points
        if (self.MST is None):
            self.compute_MST()
        
        fig = plt.figure()
        ax = fig.gca(projection = '3d')
        
        ax.set_title(title)
        ax.set_xlabel("$X [h^{-1} Mpc]$")
        ax.set_ylabel("$Y [h^{-1} Mpc]$")
        ax.set_zlabel("$Z [h^{-1} Mpc]$")
        
        # plotting nodes:
        List_indexes = np.arange(self.count)
        Indexes_kept = np.random.choice(List_indexes, (100000, ))
        X_reduced, Y_reduced, Z_reduced = [], [], []
        for i in range(100000):
            X_reduced.append(self.CM[Indexes_kept[i]][0])
            Y_reduced.append(self.CM[Indexes_kept[i]][1])
            Z_reduced.append(self.CM[Indexes_kept[i]][2])
        X_reduced = np.asarray(X_reduced)
        Y_reduced = np.asarray(Y_reduced)
        Z_reduced = np.asarray(Z_reduced)
        ax.scatter(X_reduced, Y_reduced, Z_reduced, s = 0.1, color = 'r')
        
        # plotting MST edges
        d, l, b, s, l_index, b_index = self.MST.get_stats(include_index = True)
        for i in range(np.shape(l_index)[1]):
            if ((np.isin(l_index[0][i], Indexes_kept) == True) and (np.isin(l_index[1][i], Indexes_kept) == True)):
                X = [self.CM[l_index[0][i]][0], self.CM[l_index[1][i]][0]]
                Y = [self.CM[l_index[0][i]][1], self.CM[l_index[1][i]][1]]
                Z = [self.CM[l_index[0][i]][2], self.CM[l_index[1][i]][2]]
                ax.plot3D(X, Y, Z, color = 'k', linewidth = 2)
        
        plt.show(block = True)
    
    def plot_MST_histogram(self, title = " "):
        if (self.MST_histogram is None):
            self.compute_MST_histogram() # will yield non-statistical results for ALF as default behaviour
        
        plot_histograms = mist.PlotHistMST()
        plot_histograms.read_mst(self.MST_histogram, label = self.type)
        plot_histograms.plot(figsize = (9, 4))
        
        plt.show(block = True)
    
    def save(self):
        return("save To Be Done for " + self.type)
    
    def load(self):
        return("load To Be Done for " + self.type)
    

class Catalogue_Illustris(Catalogue):
    
    def __init__(self, basePath = './Illustris-1/output/', simulation_number = 135):
        self.type = "Illustris 1"
        self.color = 'green'
        self.basePath = basePath
        self.simulation_number = simulation_number
        
        self.count = None # float -  number of galaxies
        self.box_size = None # float - Mpc / h
        self.CM = None # np.array(count, 3) - Mpc / h
        self.h = None # float - km/s / Mpc
        self.Masses = None # float - M_sun / h
        
        self.parameters_HMF = None #dict(bin_min_HMF, bin_max_HMF, n_bin_HMF) - first two in M_sun / h
        self.Bins_HMF = None # np.array(n_bin_HMF, 1)
        self.HMF = None # np.array(n_bin_HMF, 1)
        
        self.parameters_2PCF = None # dict(bin_min, bin_max, n_bin_2PCF, min_reliable, max_reliable) - first two and last two in Mpc / h
        self.Bins = None # np.array(n_bin_2PCF, 1) - Mpc / h
        self.Mean_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Std_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Bins_reliable = None # np.array(? , 1) # Mpc / h
        self.Mean_2PCF_reliable = None # np.array(?, 1)
        self.Std_2PCF_reliable = None # np.array(?, 1)
        
        self.MST = None # mist.mst
        self.MST_histogram = None # dict
    
    def initialise_data(self):
        self.count, self.box_size, self.CM, self.h, self.Masses = il.get_data(self.basePath, self.simulation_number)
    
    def compute_2PCF(self, bin_min, bin_max, n_bin_2PCF):
        if (self.CM is None):
            self.initialise_data()
        self.parameters_2PCF = {'bin_min' : bin_min, 'bin_max' : bin_max, 'n_bin_2PCF' : n_bin_2PCF, 'min_reliable' : None, 'max_reliable' : None}
        self.Bins, self.Mean_2PCF, self.Std_2PCF = il.get_2PCF(input_data = self.CM, bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF, box_size = self.box_size)
    
    def compute_MST_histogram(self):
        if (self.MST is None):
            self.compute_MST()
        
        self.MST_histogram = il.get_MST_histogram(MST = self.MST)
    

class Catalogue_Abacus(Catalogue):
    
    def __init__(self, basePath = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_00_products/AbacusCosmos_720box_00_rockstar_halos/z0.100'):
        self.type = "Abacus Rockstar"
        self.color = 'blue'
        self.basePath = basePath
        
        self.count = None # float -  number of galaxies
        self.box_size = None # float - Mpc / h
        self.CM = None # np.array(count, 3) - Mpc / h
        self.h = None # float - km/s / Mpc
        self.Masses = None # float - M_sun / h
        
        self.parameters_HMF = None #dict(bin_min_HMF, bin_max_HMF, n_bin_HMF) - first two in M_sun / h
        self.Bins_HMF = None # np.array(n_bin_HMF, 1)
        self.HMF = None # np.array(n_bin_HMF, 1)
        
        self.parameters_2PCF = None # dict(bin_min, bin_max, n_bin_2PCF, min_reliable, max_reliable) - first two and last two in Mpc / h
        self.Bins = None # np.array(n_bin_2PCF, 1) - Mpc / h
        self.Mean_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Std_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Bins_reliable = None # np.array(? , 1) # Mpc / h
        self.Mean_2PCF_reliable = None # np.array(?, 1)
        self.Std_2PCF_reliable = None # np.array(?, 1)
        
        self.MST = None # mist.mst
        self.MST_histogram = None # dict
    
    def initialise_data(self):
        self.inputPath = self.basePath + '/header'
        self.dataPath = self.basePath
        self.count, self.box_size, self.CM, self.h, self.Masses = ab.get_data(self.dataPath, self.inputPath)
    
    def compute_2PCF(self, bin_min, bin_max, n_bin_2PCF):
        if (self.CM is None):
            self.initialise_data()
        self.parameters_2PCF = {'bin_min' : bin_min, 'bin_max' : bin_max, 'n_bin_2PCF' : n_bin_2PCF, 'min_reliable' : None, 'max_reliable' : None}
        self.Bins, self.Mean_2PCF, self.Std_2PCF = ab.get_2PCF(input_data = self.CM, bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF, box_size = self.box_size)
    
    def compute_MST_histogram(self, jacknife = False):
        if (jacknife == False):
            if (self.MST is None):
                self.compute_MST()
            
            self.MST_histogram = ab.get_MST_histogram(MST = self.MST)
            
        else:
            if (self.CM is None):
                self.initialise_data()
            
            self.MST_histogram = ab.get_MST_histogram(jacknife = True, CM = sef.CM)


class Catalogue_ALF(Catalogue):
    
    def __init__(self, count, alpha, beta, gamma, t0, ts, box_size):
        self.type = 'ALF'
        self.color = 'red'
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.t0 = t0
        self.ts = ts
        
        self.count = count
        self.box_size = box_size
        self.CM = None # np.array(count, 3) - Mpc / h
        self.h = 1
        self.Masses = np.array([10**10 for i in range(count)])
        
        self.parameters_HMF = None #dict(bin_min_HMF, bin_max_HMF, n_bin_HMF) - first two in M_sun / h
        self.Bins_HMF = None # np.array(n_bin_HMF, 1)
        self.HMF = None # np.array(n_bin_HMF, 1)
        
        self.parameters_2PCF = None # dict(bin_min, bin_max, n_bin_2PCF, min_reliable, max_reliable) - first two and last two in Mpc / h
        self.Bins = None # np.array(n_bin_2PCF, 1) - Mpc / h
        self.Mean_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Std_2PCF = None # np.array(n_bin_2PCF, 1)
        self.Bins_reliable = None # np.array(? , 1) # Mpc / h
        self.Mean_2PCF_reliable = None # np.array(?, 1)
        self.Std_2PCF_reliable = None # np.array(?, 1)
        
        self.MST = None # mist.mst
        self.MST_histogram = None # dict
    
    def initialise_data(self):
        self.CM = alf.get_data(count = self.count, alpha = self.alpha, beta = self.beta, gamma = self.gamma, t0 = self.t0, ts = self.ts, box_size = self.box_size, mode = '3D')
    
    def compute_2PCF(self, bin_min, bin_max, n_bin_2PCF):
        if (self.CM is None):
            self.initialise_data()
        
        self.parameters_2PCF = {'bin_min' : bin_min, 'bin_max' : bin_max, 'n_bin_2PCF' : n_bin_2PCF, 'min_reliable' : None, 'max_reliable' : None}
        self.Bins, self.Mean_2PCF, self.Std_2PCF = alf.get_2PCF(bin_min = bin_min, bin_max = bin_max, n_bin_2PCF = n_bin_2PCF, box_size = self.box_size, count = self.count, alpha = self.alpha, beta = self.beta, gamma = self.gamma, t0 = self.t0, ts = self.ts, mode = '3D')
    
    def compute_MST_histogram(self, mode_MST = 'SingleMST'): # Can be either for the current ALF distribution, or for the statistical average ALF with the same parameters
        if (mode_MST == 'SingleMST'):
            if (self.MST is None):
                self.compute_MST()
        
            self.MST_histogram = alf.get_MST_histogram(MST = self.MST, mode_MST = 'SingleMST')
        else:
            if (self.MST is None):
                self.compute_MST()
            
            self.MST_histogram = alf.get_MST_histogram(mode_MST = mode_MST, count = self.count, alpha = self.alpha, beta = self.beta, gamma = self.gamma, t0 = self.t0, ts = self.ts, box_size = self.box_size, mode = '3D')


## Tools

def compare_HMFs(List_catalogues, title = " ", figure = None): # we assume all catalogues HMF have been computed already
    n = len(List_catalogues)
    for i in range(n):
        if (List_catalogues[i].HMF is None):
            return("HMFs should be computed before comparison. This is not true for element in position " + i + "in List_catalogues")
    
    plot_at_the_end = False
    if (figure is None):
        	misleading = plt.figure()
        	figure = misleading.gca()
        	plot_at_the_end = True
    
    figure.set_title(title)
    figure.set_title(title)
    figure.set_xlabel("Mass [$M_{\odot}$]")
    figure.set_ylabel("Count")
    figure.set_xscale('log')
    figure.set_yscale('log')
    
    for catalogue in List_catalogues:
        plt.plot(catalogue.Bins_HMF[:-1], catalogue.HMF, color = catalogue.color, label = catalogue.type)
    
    figure.legend()
    
    if plot_at_the_end:
    	  plt.show(block = True)
    
def compare_2PCFs(List_catalogues, title = " ", figure = None): # we assume all catalogues 2PCFs have been computed already, and we only provide support for the full 2PCF. Please zoom on the interesting part of the graph if necessary.
    n = len(List_catalogues)
    for i in range(n):
        if (List_catalogues[i].Mean_2PCF is None):
            return("2PCFs should be computed before comparison. This is not true for element in position " + i + "in List_catalogues")
    
    plot_at_the_end = False
    if (figure is None):
    	  misleading = plt.figure()
    	  figure = misleading.gca()
    	  plot_at_the_end = True
    
    figure.set_title(title)
    figure.set_xlabel("$r [h^{-1} Mpc]$")
    figure.set_ylabel("$\\xi(r)$")
    figure.set_xscale('log')
    figure.set_yscale('log')
    
    for catalogue in List_catalogues:
        figure.plot(catalogue.Bins, catalogue.Mean_2PCF, catalogue.color, label = catalogue.type)
        figure.plot(catalogue.Bins, catalogue.Mean_2PCF - catalogue.Std_2PCF, color = catalogue.color, linestyle = '--')
        figure.plot(catalogue.Bins, catalogue.Mean_2PCF + catalogue.Std_2PCF, color = catalogue.color, linestyle = '--')
    
    figure.legend()
    
    if plot_at_the_end:
    	plt.show(block = True)

def compare_MST_histograms(List_catalogues, usemean = True, whichcomp = None, title = " ", saveas = None): # we assume all catalogues MST histograms have been computed already
    n = len(List_catalogues)
    for i in range(n):
        if (List_catalogues[i].MST_histogram is None):
            return("MST histograms should be computed before comparison. This is not true for element in position " + i + "in List_catalogues")
    
    plot_histograms = mist.PlotHistMST()
    for catalogue in List_catalogues:
        MST_histogram = catalogue.MST_histogram
        plot_histograms.read_mst(MST_histogram, label = catalogue.type)
    plot_histograms.plot(usecomp = True, usemean = usemean, whichcomp = whichcomp, figsize = (9, 6), saveas = saveas)
