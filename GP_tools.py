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
    
    def __init__(self, X, Y, N_points_per_simu = [5, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated"):
        self.X_d_data, self.X_l_data, self.X_b_data, self.X_s_data = X # (n_simu * n_d/b/l/s_points_per_simu, 6) arrays
        self.Y_d_data, self.Y_l_data, self.Y_b_data, self.Y_s_data = Y # (n_simu * n_d/b/l/s_points_per_simu, 1) arrays
        
        N = [np.shape(self.X_d_data)[0], np.shape(self.X_l_data)[0], np.shape(self.X_b_data)[0], np.shape(self.X_s_data)[0]]
        self.n_d_points, self.n_l_points, self.n_b_points, self.n_s_points = N # ints
        self.n_d_points_per_simu, self.n_l_points_per_simu, self.n_b_points_per_simu, self.n_s_points_per_simu = N_points_per_simu # ints
        self.n_simu = self.n_d_points / self.n_d_points_per_simu # int
        
        self.noise_d, self.noise_l, self.noise_b, self.noise_s = Noise # floats
        
        self.Kernels = None # list of GPy.kern objects
        self.type_kernel = type_kernel
        self.make_kernels()
        
        self.Models = None # list of GPy.model objects
        self.make_models()
    
    def make_kernels(self):
        if (self.type_kernel == "Separated"):
            kernel_d = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
            kernel_l = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
            kernel_b = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
            kernel_s = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
            self.Kernels = [kernel_d, kernel_l, kernel_b, kernel_s]
            
        elif (self.type_kernel == "Coregionalized"):
            kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
            kernel = GPy.util.multioutput.ICM(input_dim = 6, num_outputs = 4, W_rank = 4, kernel = kernel_input) # rank 4 since there are 4 outputs
            self.Kernels = [kernel]
        
        else:
            print("Error : kernel type not recognized when initialising the GP")
            
    def make_models(self):
        if (self.type_kernel == "Separated"):
            gp_d = GPy.models.GPRegression(self.X_d_data, self.Y_d_data, self.Kernels[0])
            gp_l = GPy.models.GPRegression(self.X_l_data, self.Y_l_data, self.Kernels[1])
            gp_b = GPy.models.GPRegression(self.X_b_data, self.Y_b_data, self.Kernels[2])
            gp_s = GPy.models.GPRegression(self.X_s_data, self.Y_s_data, self.Kernels[3])
            
            if (self.noise_d is not None):
                gp_d.Gaussian_noise.variance = self.noise_d
                gp_d.Gaussian_noise.variance.fix()
            if (self.noise_l is not None):
                gp_l.Gaussian_noise.variance = self.noise_l
                gp_l.Gaussian_noise.variance.fix()
            if (self.noise_b is not None):
                gp_b.Gaussian_noise.variance = self.noise_b
                gp_b.Gaussian_noise.variance.fix()
            if (self.noise_s is not None):
                gp_s.Gaussian_noise.variance = self.noise_s
                gp_s.Gaussian_noise.variance.fix()
            
            self.Models = [gp_d, gp_l, gp_b, gp_s]
        
        elif (self.type_kernel == "Coregionalized"):
            gp = GPy.models.GPCoregionalizedRegression([self.X_d_data, self.X_l_data, self.X_b_data, self.X_s_data], [self.Y_d_data, self.Y_l_data, self.Y_b_data, self.Y_s_data], kernel = self.Kernels[0])
            
            if (self.noise_d is not None):
                gp.mixed_noise.Gaussian_noise_0.variance = self.noise_d
                gp.mixed_noise.Gaussian_noise_0.variance.fix()
            if (self.noise_l is not None):
                gp.mixed_noise.Gaussian_noise_1.variance = self.noise_l
                gp.mixed_noise.Gaussian_noise_1.variance.fix()
            if (self.noise_b is not None):
                gp.mixed_noise.Gaussian_noise_2.variance = self.noise_b
                gp.mixed_noise.Gaussian_noise_2.variance.fix()
            if (self.noise_s is not None):
                gp.mixed_noise.Gaussian_noise_3.variance = self.noise_s
                gp.mixed_noise.Gaussian_noise_3.variance.fix()
            
            self.Models = [gp]
    
    def change_kernels(self, New_kernels, Stats = None):
        if (self.type_kernel == "Separated"):
            for i in range(len(Stats)):
                self.Kernels[Stats[i]] = New_kernels[i]
            self.make_models()
        
        elif (self.type_kernel == "Coregionalized"):
            kernel_input = New_kernels[0]
            kernel = GPy.util.multioutput.ICM(input_dim = 6, num_outputs = 4, W_rank = 4, kernel = kernel_input) # rank 4 since there are 4
            self.Kernels[0] = kernel
    
    def optimize_models(self, optimizer = 'lbfgsb'):
        if (self.type_kernel == "Separated"):
            self.Models[0].optimize(optimizer = optimizer)
            self.Models[1].optimize(optimizer = optimizer)
            self.Models[2].optimize(optimizer = optimizer)
            self.Models[3].optimize(optimizer = optimizer)
        
        elif (self.type_kernel == "Coregionalized"):
            self.Models[0].optimize(optimizer = optimizer)
    
    def print_models(self):
        if (self.type_kernel == "Separated"):
            for i in range(4):
                gp = self.Models[i]
                Stats = ['d', 'l', 'b', 's']
                
                print("GP model for " + str(Stats[i]) + " :")
                print(gp.rbf.variance)
                print(gp.rbf.lengthscale)
                print(gp.Gaussian_noise.variance)
                print("Obj : " + str(gp.objective_function()))
                print("\n")
        
        elif (self.type_kernel == "Coregionalized"):
            gp = self.Models[0]
            
            print("GP Coregionalized model :")
            print(gp.ICM.rbf.variance)
            print(gp.ICM.rbf.lengthscale)
            print(gp.ICM.B.W)
            print(gp.ICM.B.kappa)
            print(gp.mixed_noise.Gaussian_noise_0.variance)
            print(gp.mixed_noise.Gaussian_noise_1.variance)
            print(gp.mixed_noise.Gaussian_noise_2.variance)
            print(gp.mixed_noise.Gaussian_noise_3.variance)
            print("Obj : " + str(gp.objective_function()))
            print("\n")
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_cosmologies = np.shape(X_new)[0]
        
        if (self.type_kernel == "Separated"):
            X_predicted, Y_predicted, Cov = [], [], []
            for i in range(n_new_cosmologies):
                h0, w0, ns, sigma8, omegaM = X_new[i]
                
                X_d = self.X_d_data[0 : self.n_d_points_per_simu, 5]
                X_l = self.X_l_data[0 : self.n_l_points_per_simu, 5]
                X_b = self.X_b_data[0 : self.n_b_points_per_simu, 5]
                X_s = self.X_s_data[0 : self.n_s_points_per_simu, 5]
                X_d_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d] for x_d in X_d], (self.n_d_points_per_simu, 6))
                X_l_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l] for x_l in X_l], (self.n_l_points_per_simu, 6))
                X_b_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b] for x_b in X_b], (self.n_b_points_per_simu, 6))
                X_s_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s] for x_s in X_s], (self.n_s_points_per_simu, 6))
                
                Y_d_predicted, Cov_d = self.Models[0].predict(X_d_new, full_cov = True)
                Y_l_predicted, Cov_l = self.Models[1].predict(X_l_new, full_cov = True)
                Y_b_predicted, Cov_b = self.Models[2].predict(X_b_new, full_cov = True)
                Y_s_predicted, Cov_s = self.Models[3].predict(X_s_new, full_cov = True)
                
                X_predicted.append([X_d_new, X_l_new, X_b_new, X_s_new])
                Y_predicted.append([Y_d_predicted, Y_l_predicted, Y_b_predicted, Y_s_predicted])
                Cov.append([Cov_d, Cov_l, Cov_b, Cov_s])
            return(X_predicted, Y_predicted, Cov)
        
        elif (self.type_kernel == "Coregionalized"):
            X_predicted, Y_predicted, Cov = [], [], []
            for i in range(n_new_cosmologies):
                h0, w0, ns, sigma8, omegaM = X_new[i]
                
                X_d = self.X_d_data[0 : self.n_d_points_per_simu, 5]
                X_l = self.X_l_data[0 : self.n_l_points_per_simu, 5]
                X_b = self.X_b_data[0 : self.n_b_points_per_simu, 5]
                X_s = self.X_s_data[0 : self.n_s_points_per_simu, 5]
                X_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d] for x_d in X_d], (self.n_d_points_per_simu, 6))
                X_l = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l] for x_l in X_l], (self.n_l_points_per_simu, 6))
                X_b = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b] for x_b in X_b], (self.n_b_points_per_simu, 6))
                X_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s] for x_s in X_s], (self.n_s_points_per_simu, 6))
                X_d_new = np.concatenate((X_d, np.array([[0] for i in range(self.n_d_points_per_simu)])), 1)
                X_l_new = np.concatenate((X_l, np.array([[1] for i in range(self.n_l_points_per_simu)])), 1)
                X_b_new = np.concatenate((X_b, np.array([[2] for i in range(self.n_b_points_per_simu)])), 1)
                X_s_new = np.concatenate((X_s, np.array([[3] for i in range(self.n_s_points_per_simu)])), 1)
                
                X = np.concatenate((X_d_new, X_l_new, X_b_new, X_s_new), 0)
                Metadata = np.concatenate((np.asarray([0 for i in range(self.n_d_points_per_simu)]), np.asarray([1 for i in range(self.n_l_points_per_simu)]), np.asarray([2 for i in range(self.n_b_points_per_simu)]), np.asarray([3 for i in range(self.n_s_points_per_simu)])))
                
                Y, C = self.Models[0].predict(X, Y_metadata={'output_index' : Metadata}, full_cov = True)
                
                Y_d_predicted = Y[0 : self.n_d_points_per_simu]
                Y_l_predicted = Y[self.n_d_points_per_simu : self.n_d_points_per_simu + self.n_l_points_per_simu]
                Y_b_predicted = Y[self.n_d_points_per_simu + self.n_l_points_per_simu : self.n_d_points_per_simu + self.n_l_points_per_simu + self.n_b_points_per_simu]
                Y_s_predicted = Y[self.n_d_points_per_simu + self.n_l_points_per_simu + self.n_b_points_per_simu:]
                
                X_predicted.append([X_d, X_l, X_b, X_s])
                Y_predicted.append([Y_d_predicted, Y_l_predicted, Y_b_predicted, Y_s_predicted])
                Cov.append(C)
            return(X_predicted, Y_predicted, Cov)
    
    def test_rms(self, X_test, Y_test):
        if (self.type_kernel == "Separated"):
            X_d_test, X_l_test, X_b_test, X_s_test = X_test
            Y_d_test, Y_l_test, Y_b_test, Y_s_test = Y_test
            
            s = 0
            for i in range(np.shape(X_d_test)[0]):
                X_d_new = np.reshape(X_d_test[i], (1, 6))
                X_l_new = np.reshape(X_l_test[i], (1, 6))
                X_b_new = np.reshape(X_b_test[i], (1, 6))
                X_s_new = np.reshape(X_s_test[i], (1, 6))
                
                y_d_predicted = (self.Models[0].predict(X_d_new, full_cov = True))[0][0][0]
                y_l_predicted = (self.Models[1].predict(X_l_new, full_cov = True))[0][0][0]
                y_b_predicted = (self.Models[2].predict(X_b_new, full_cov = True))[0][0][0]
                y_s_predicted = (self.Models[3].predict(X_s_new, full_cov = True))[0][0][0]
                
                y_d_expected = Y_d_test[i][0]
                y_l_expected = Y_l_test[i][0]
                y_b_expected = Y_b_test[i][0]
                y_s_expected = Y_s_test[i][0]
                
                s += (y_d_predicted - y_d_expected)**2 + (y_l_predicted - y_l_expected)**2 + (y_b_predicted - y_b_expected)**2 + (y_s_predicted - y_s_expected)**2
            
            ms = s / (4 * np.shape(X_d_test)[0])
            rms = np.sqrt(ms)
            return(rms)
        
        elif (self.type_kernel == "Coregionalized"):
            X_d_test, X_l_test, X_b_test, X_s_test = X_test
            Y_d_test, Y_l_test, Y_b_test, Y_s_test = Y_test
            
            s = 0
            for i in range(np.shape(X_d_test)[0]):
                X_d_new = np.reshape(X_d_test[i], (1, 6))
                X_d_new = np.concatenate((X_d_new, np.array([[0]])), 1)
                Metadata = np.asarray([0])
                
                y_d_predicted, C = self.Models[0].predict(X_d_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_d_expected = Y_d_test[i][0]
                
                s += (y_d_predicted - y_d_expected)**2
            for i in range(np.shape(X_l_test)[0]):
                X_l_new = np.reshape(X_l_test[i], (1, 6))
                X_l_new = np.concatenate((X_l_new, np.array([[1]])), 1)
                Metadata = np.asarray([1])
                
                y_l_predicted, C = self.Models[0].predict(X_l_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_l_expected = Y_l_test[i][0]
                
                s += (y_l_predicted - y_l_expected)**2
            for i in range(np.shape(X_b_test)[0]):
                X_b_new = np.reshape(X_b_test[i], (1, 6))
                X_b_new = np.concatenate((X_b_new, np.array([[2]])), 1)
                Metadata = np.asarray([2])
                
                y_b_predicted, C = self.Models[0].predict(X_b_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_b_expected = Y_b_test[i][0]
                
                s += (y_b_predicted - y_b_expected)**2
            for i in range(np.shape(X_s_test)[0]):
                X_s_new = np.reshape(X_s_test[i], (1, 6))
                X_s_new = np.concatenate((X_s_new, np.array([[3]])), 1)
                Metadata = np.asarray([3])
                
                y_s_predicted, C = self.Models[0].predict(X_s_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_s_expected = Y_s_test[i][0]
                
                s += (y_s_predicted - y_s_expected)**2
            
            ms = s / (np.shape(X_d_test)[0] + np.shape(X_l_test)[0] + np.shape(X_b_test)[0] + np.shape(X_s_test)[0])
            rms = np.sqrt(ms)
            return(rms)
    
    def test_chi2(self, X_test, Y_test):
        if (self.type_kernel == "Separated"):
            X_d_test, X_l_test, X_b_test, X_s_test = X_test
            Y_d_test, Y_l_test, Y_b_test, Y_s_test = Y_test
            
            s = 0
            for i in range(np.shape(X_d_test)[0]):
                X_d_new = np.reshape(X_d_test[i], (1, 6))
                X_l_new = np.reshape(X_l_test[i], (1, 6))
                X_b_new = np.reshape(X_b_test[i], (1, 6))
                X_s_new = np.reshape(X_s_test[i], (1, 6))
                
                y_d_predicted, C_d = (self.Models[0].predict(X_d_new, full_cov = True))
                y_l_predicted, C_l = (self.Models[1].predict(X_l_new, full_cov = True))
                y_b_predicted, C_b = (self.Models[2].predict(X_b_new, full_cov = True))
                y_s_predicted, C_s = (self.Models[3].predict(X_s_new, full_cov = True))
                
                y_d_expected = Y_d_test[i][0]
                y_l_expected = Y_l_test[i][0]
                y_b_expected = Y_b_test[i][0]
                y_s_expected = Y_s_test[i][0]
                
                s += (y_d_predicted - y_d_expected)**2 / (C_d + self.Models[0].Gaussian_noise.variance**2)
                s += (y_l_predicted - y_l_expected)**2 / (C_b + self.Models[1].Gaussian_noise.variance**2)
                s += (y_b_predicted - y_b_expected)**2 / (C_l + self.Models[2].Gaussian_noise.variance**2)
                s += (y_s_predicted - y_s_expected)**2 / (C_s + self.Models[3].Gaussian_noise.variance**2)
            
            ms = s / (4 * np.shape(X_d_test)[0])
            chi_2 = np.sqrt(ms)
            return(chi_2)
        
        elif (self.type_kernel == "Coregionalized"):
            X_d_test, X_l_test, X_b_test, X_s_test = X_test
            Y_d_test, Y_l_test, Y_b_test, Y_s_test = Y_test
            
            s = 0
            for i in range(np.shape(X_d_test)[0]):
                X_d_new = np.reshape(X_d_test[i], (1, 6))
                X_d_new = np.concatenate((X_d_new, np.array([[0]])), 1)
                Metadata = np.asarray([0])
                
                y_d_predicted, C = self.Models[0].predict(X_d_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_d_expected = Y_d_test[i][0]
                
                s += (y_d_predicted - y_d_expected)**2 / (C + self.Models[0].mixed_noise.Gaussian_noise_0.variance**2)
            for i in range(np.shape(X_l_test)[0]):
                X_l_new = np.reshape(X_l_test[i], (1, 6))
                X_l_new = np.concatenate((X_l_new, np.array([[1]])), 1)
                Metadata = np.asarray([1])
                
                y_l_predicted, C = self.Models[0].predict(X_l_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_l_expected = Y_l_test[i][0]
                
                s += (y_l_predicted - y_l_expected)**2 / (C + self.Models[0].mixed_noise.Gaussian_noise_1.variance**2)
            for i in range(np.shape(X_b_test)[0]):
                X_b_new = np.reshape(X_b_test[i], (1, 6))
                X_b_new = np.concatenate((X_b_new, np.array([[2]])), 1)
                Metadata = np.asarray([2])
                
                y_b_predicted, C = self.Models[0].predict(X_b_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_b_expected = Y_b_test[i][0]
                
                s += (y_b_predicted - y_b_expected)**2 / (C + self.Models[0].mixed_noise.Gaussian_noise_2.variance**2)
            for i in range(np.shape(X_s_test)[0]):
                X_s_new = np.reshape(X_s_test[i], (1, 6))
                X_s_new = np.concatenate((X_s_new, np.array([[3]])), 1)
                Metadata = np.asarray([3])
                
                y_s_predicted, C = self.Models[0].predict(X_s_new, Y_metadata={'output_index' : Metadata}, full_cov = True)
                y_s_expected = Y_s_test[i][0]
                
                s += (y_s_predicted - y_s_expected)**2 / (C + self.Models[0].mixed_noise.Gaussian_noise_3.variance**2)
            
            ms = s / (np.shape(X_d_test)[0] + np.shape(X_l_test)[0] + np.shape(X_b_test)[0] + np.shape(X_s_test)[0])
            chi_2 = np.sqrt(ms)
            return(chi_2)
    
    def chi_2(self, Y_model, Noise_model, Y_observation, Noise_observation):
        if (self.type_kernel == "Separated"):
            s = 0
            
            n_simulations = np.shape(Y_observation)[0]
            
            for i in range(n_simulations):
                Sigma_model = Noise_model[i]
                if (Sigma_model is None):
                    Diag_d = [self.Models[0].Gaussian_noise.variance[0] for j in range(self.n_d_points_per_simu)]
                    Diag_l = [self.Models[1].Gaussian_noise.variance[0] for j in range(self.n_l_points_per_simu)]
                    Diag_b = [self.Models[2].Gaussian_noise.variance[0] for j in range(self.n_b_points_per_simu)]
                    Diag_s = [self.Models[3].Gaussian_noise.variance[0] for j in range(self.n_s_points_per_simu)]
                    Diag = Diag_d + Diag_l + Diag_b + Diag_s
                    Sigma_model = np.diag(Diag)
                
                Sigma_obs = linalg.block_diag(Noise_observation[i][0], Noise_observation[i][1], Noise_observation[i][2], Noise_observation[i][3])
                Sigma = Sigma_model + Sigma_obs
                Sigma_inv = np.linalg.inv(Sigma)
                
                U = np.concatenate((Y_model[i][0], Y_model[i][1], Y_model[i][2], Y_model[i][3]), 0)
                V = np.concatenate((Y_observation[i][0], Y_observation[i][1], Y_observation[i][2], Y_observation[i][3]), 0)
                
                s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
            
            ms = s / (4 * n_simulations)
            chi_2 = np.sqrt(ms)
            return(chi_2)
        
        elif (self.type_kernel == "Coregionalized"):
            return("Error : chi_2 has not yet been written for coregionalized kernels")
    