## Imports
import numpy as np
import scipy.spatial as spatial
import scipy.linalg as linalg
import GPy as GPy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

## GP Class
class GP():
    
    def __init__(self, X, Y, N_points_per_simu = [4, 5], Noise = [None, None]):
        self.n_d_points_per_simu, self.n_s_points_per_simu = N_points_per_simu # ints
        self.n_d_points, self.n_s_points = np.shape(X[0])[0], np.shape(X[1])[0] # ints
        self.n_simu = self.n_d_points / self.n_d_points_per_simu # int
                
        self.X_d_data, self.X_s_data = X # (n_simu * n_d/b/l/s_points_per_simu, 6) arrays
        #self.X_d_data = np.concatenate((self.X_d_data, np.array([[0] for i in range(self.n_d_points)])), 1)
        #self.X_s_data = np.concatenate((self.X_s_data, np.array([[1] for i in range(self.n_s_points)])), 1)
            
        self.Y_d_data, self.Y_s_data = Y # (n_simu * n_d/b/l/s_points_per_simu, 1) arrays
        
        self.noise_d, self.noise_s = Noise # None or (1, n_d/l/b/s_points) arrays
        
        self.Kernels = None # list of GPy.kern objects
        self.make_kernels()
        
        self.Models = None # list of GPy.model objects
        self.make_models()
    
    def make_kernels(self):
        kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
        kernel = GPy.util.multioutput.ICM(input_dim = 6, num_outputs = 2, W_rank = 2, kernel = kernel_input)
        
        self.Kernels = [kernel]
            
    def make_models(self):
        gp = GPy.models.GPCoregionalizedRegression([self.X_d_data, self.X_s_data], [self.Y_d_data, self.Y_s_data], kernel = self.Kernels[0])
        
        if (self.noise_d is not None):
            gp.mixed_noise.Gaussian_noise_0.variance = self.noise_d
            gp.mixed_noise.Gaussian_noise_0.variance.fix()
        if (self.noise_s is not None):
            gp.mixed_noise.Gaussian_noise_3.variance = self.noise_s
            gp.mixed_noise.Gaussian_noise_3.variance.fix()
        
        self.Models = [gp]
    
    def change_kernels(self, New_kernel):
        kernel = New_kernels
        kernel = GPy.util.multioutput.ICM(input_dim = 6, num_outputs = 2, W_rank = 2, kernel = kernel)
        
        self.Kernels[0] = kernel
    
    def change_models_to_heteroscedatic(self, Noise = [None, None]):
        print("Error :  Heteroscedacity has not yet been implemented for coregionalized kernels")
    
    def optimize_models(self, optimizer = 'lbfgsb'):
        self.Models[0].optimize(optimizer = optimizer)
    
    def print_models(self):
        gp = self.Models[0]
        
        print("GP Coregionalized model :")
        print(gp.ICM.rbf.variance)
        print(gp.ICM.rbf.lengthscale)
        print(gp.ICM.B.W)
        print(gp.ICM.B.kappa)
        print(gp.mixed_noise.Gaussian_noise_0.variance)
        print(gp.mixed_noise.Gaussian_noise_1.variance)
        print("Obj : " + str(gp.objective_function()))
        print("\n")
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_cosmologies = np.shape(X_new)[0]
        
        X_d_predicted, Y_d_predicted, Cov_d_predicted = [], [], []
        X_s_predicted, Y_s_predicted, Cov_s_predicted = [], [], []
        for i in range(n_new_cosmologies):
            h0, w0, ns, sigma8, omegaM = X_new[i]
            
            X_d = self.X_d_data[0 : self.n_d_points_per_simu, 5]
            X_s = self.X_s_data[0 : self.n_s_points_per_simu, 5]
            
            X_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d] for x_d in X_d], (self.n_d_points_per_simu, 6))
            X_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s] for x_s in X_s], (self.n_s_points_per_simu, 6))
            
            X_d_new = np.concatenate((X_d, np.array([[0] for i in range(self.n_d_points_per_simu)])), 1)
            X_s_new = np.concatenate((X_s, np.array([[1] for i in range(self.n_s_points_per_simu)])), 1)
            
            Y_d_new, Cov_d_new = self.Models[0].predict(X_d_new, Y_metadata={'output_index' : np.asarray([0 for i in range(self.n_d_points_per_simu)])}, full_cov = True)
            Y_s_new, Cov_s_new = self.Models[0].predict(X_s_new, Y_metadata={'output_index' : np.asarray([1 for i in range(self.n_d_points_per_simu)])}, full_cov = True)
            
            X_d_predicted.append(X_d_new)
            Y_d_predicted.append(Y_d_new)
            Cov_d_predicted.append(Cov_d_new)
            X_s_predicted.append(X_s_new)
            Y_s_predicted.append(Y_s_new)
            Cov_s_predicted.append(Cov_s_new)
        
        return(X_d_predicted, Y_d_predicted, Cov_d_predicted, X_s_predicted, Y_s_predicted, Cov_s_predicted)
    
    def compute_chi2_test(self, X_test, Y_test, Y_std_test, metadata): # we assume the model either has already been optimized
        n_test = np.shape(X_test)[0]
        
        s = 0
        for i in range(n_test):
            x_test = np.reshape(X_test[i], (1, 6))
            x_test = np.concatenate((x_test, np.asarray([[metadata[i]]])), 1)
            y_predicted, std_test = self.Models[0].predict(x_test, full_cov = True, Y_metadata = {'output_index' : np.asarray(metadata[i])})
            y_expected, y_std_expected = Y_test[i], Y_std_test[i]
            s += (y_predicted - y_expected)**2 / y_std_expected**2

        return(np.asscalar(s))
    
    def compute_ms_test(self, X_test, Y_test, Y_std_test, metadata): # we assume the model either has already been optimized
        n_test = np.shape(X_test)[0]
        
        s = 0
        for i in range(n_test):
            x_test = np.reshape(X_test[i], (1, 6))
            x_test = np.concatenate((x_test, np.asarray([[metadata[i]]])), 1)
            y_predicted, std_test = self.Models[0].predict(x_test, full_cov = True, Y_metadata = {'output_index' : np.asarray(metadata[i])})
            y_expected, y_std_expected = Y_test[i], Y_std_test[i]
            s += (y_predicted - y_expected)**2

        return(np.asscalar(s))