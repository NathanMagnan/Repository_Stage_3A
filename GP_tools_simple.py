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
    
    def __init__(self, X, Y, n_points_per_simu = 5, Noise = None, make_covariance_matrix = True):
        self.X_data = X # (n_simu * n_points_per_simu, 6) array
        self.Y_data = Y # (n_simu * n_points_per_simu, 1) array
        
        self.n_points = np.shape(self.X_data)[0] # ints
        self.n_points_per_simu = n_points_per_simu # ints
        self.n_simu = self.n_points / self.n_points_per_simu # int
        
        self.Noise = Noise # None or (1, n_points) array
        
        self.kernel = None # GPy.kern object
        self.make_kernel()
        
        self.model = None # GPy.model object
        self.make_model()
        
        self.cov = None # (n_points_per_simu, n_points_per_simu) array
        if make_covariance_matrix:
            self.make_covariance_matrix()
    
    def make_kernel(self):
        self.kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
    
    def make_model(self):
        gp = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.Noise is not None):
            gp = GPy.models.GPHeteroscedasticRegression(self.X_data, self.Y_data, self.kernel)
            gp['.*het_Gauss.variance'] = abs(self.Noise)
            gp.het_Gauss.variance.fix()
        
        self.model = gp
    
    def make_covariance_matrix(self):
        # defining the groups
        n_groups = int(self.n_simu)
        
        List_groups = []
        for i in range(n_groups):
            start_group = ((i * int(self.n_simu)) // n_groups)
            end_group =  (((i + 1) * int(self.n_simu)) // n_groups)
            
            X_test_a = self.X_data[start_group * self.n_points_per_simu : end_group * self.n_points_per_simu]
            X_data_a = np.concatenate((self.X_data[0 : start_group * self.n_points_per_simu], self.X_data[end_group * self.n_points_per_simu :]), 0)
            Y_test_a = self.Y_data[start_group * self.n_points_per_simu : end_group * self.n_points_per_simu]
            Y_data_a = np.concatenate((self.Y_data[0 : start_group * self.n_points_per_simu], self.Y_data[end_group * self.n_points_per_simu :]), 0)
            
            List_groups.append((X_data_a, Y_data_a, X_test_a, Y_test_a))
        
        # Evaluating the GP's errors
        Errors = []
        
        for j in range(n_groups):
            # getting the right kernel
            kernel_a = self.kernel.copy()
            
            # getting the right data and test groups
            X_data_a, Y_data_a, X_test_a, Y_test_a = List_groups[j]
            
            # creating the gaussian process and optimizing it
            gp_a = GP(X = X_data_a, Y = Y_data_a, n_points_per_simu = self.n_points_per_simu, Noise = None, make_covariance_matrix = False)
            gp_a.change_kernel(new_kernel = kernel_a, make_covariance_matrix = False)
            gp_a.optimize_model(optimizer = 'lbfgsb')
            
            # getting the errors
            errors = gp_a.compute_error_test(X_test_a, Y_test_a)
            
            # adding the errors to the lsit
            Errors.append(errors)
        
        # computing the covariance matrix from the error
        Errors = np.asarray(Errors).T
        
        cov = [[0 for i in range(self.n_points_per_simu)] for j in range(self.n_points_per_simu)]
        for i in range(self.n_points_per_simu):
            for j in range(self.n_points_per_simu):
                sum = 0
                for k in range(n_groups):
                    sum += Errors[i, k] * Errors[j, k]
                cov[i][j] = sum / (n_groups - 1)
        cov = np.asarray(cov)
        
        # saving the covariance matrix
        self.cov = cov
    
    def change_kernel(self, new_kernel, make_covariance_matrix = True):
        self.kernel = new_kernel
        self.make_model()
        if make_covariance_matrix :
            self.make_covariance_matrix()
    
    def change_model_to_heteroscedatic(self, Noise = None):
        self.Noise = Noise # None or (1, n_points) array
        
        gp = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.Noise is not None):
            gp = GPy.models.GPHeteroscedasticRegression(self.X_data, self.Y_data, self.kernel)
            gp['.*het_Gauss.variance'] = abs(self.Noise)
            gp.het_Gauss.variance.fix()
        
        self.model = gp
        self.make_covariance_matrix()
    
    def optimize_model(self, optimizer = 'lbfgsb'):
        self.model.optimize(optimizer = optimizer)
    
    def print_model(self):
        gp = self.model
        
        print("GP model :")
        print(gp.rbf.variance)
        print(gp.rbf.lengthscale)
        print("GP covariance matrix : ")
        print(self.cov)
        print("Obj : " + str(gp.objective_function()))
        print("\n")
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_cosmologies = np.shape(X_new)[0]
        
        X_predicted, Y_predicted, Cov = [], [], []
        for i in range(n_new_cosmologies):
            h0, w0, ns, sigma8, omegaM = X_new[i]
            
            X = self.X_data[-self.n_points_per_simu:, 5]
            X_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x] for x in X], (self.n_points_per_simu, 6))
            
            Y_new, Cov_new = self.model.predict(X_new, full_cov = True, Y_metadata = {'output_index' : np.array([0])})
            
            X_predicted.append(X_new)
            Y_predicted.append(Y_new)
            Cov.append(self.cov)
        return(X_predicted, Y_predicted, Cov)
    
    def compute_error_test(self, X_test, Y_test): # we assume the model either has already been optimized
        n_test = np.shape(X_test)[0]
        
        errors = []
        for i in range(n_test):
            x_test = np.reshape(X_test[i], (1, 6))
            y_test, std_test = self.model.predict(x_test, full_cov = True, Y_metadata = {'output_index' : np.array([0])})
            errors.append(np.asscalar(y_test - Y_test[i]))

        return(errors)
    
    def likelihood_ms(self, Y_model, Y_observation, Noise_model = None, Noise_observation = None):
        s = 0
        
        n_simulations = np.shape(Y_observation)[0]
        
        for i in range(n_simulations):
            Sigma_inv = np.identity(self.n_points_per_simu)
            
            U = Y_model[i]
            V = Y_observation[i]
            
            s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
        
        ms = s / (self.n_points_per_simu * n_simulations)
        return(ms)
    
    def likelihood_chi2_bd(self, Y_model, Noise_model, Y_observation, Noise_observation):
        s = 0
        
        n_simulations = np.shape(Y_observation)[0]
        
        for i in range(n_simulations):
            Sigma_model = Noise_model[i]
            Sigma_observation = np.diagflat(Noise_observation[i]**2)
            
            #Sigma = Sigma_model + Sigma_observation
            Sigma = 2 * Sigma_observation
            Sigma_inv = np.linalg.inv(Sigma)
            
            U = Y_model[i]
            V = Y_observation[i]
            W = U - V
            
            s += np.dot(np.dot(np.transpose(W), Sigma_inv), W)
            #s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
        
        chi2 = s / (n_simulations)
        return(np.asscalar(chi2))
    
    def likelihood_chi2_ad(self, Y_model, Noise_model, Y_observation, Noise_observation, N = 21):
        s = 0
        
        n_simulations = np.shape(Y_observation)[0]
        
        for i in range(n_simulations):
            Sigma_model = Noise_model[i]
            Sigma_observation = np.diagflat(Noise_observation[i]**2)
            
            #Sigma = Sigma_model + Sigma_observation
            Sigma = 2 * Sigma_observation
            Sigma_inv = np.linalg.inv(Sigma)
            
            U = Y_model[i]
            V = Y_observation[i]
            W = U - V
            
            det = np.linalg.det(Sigma)
            
            #s += -2 * np.log(det**(-0.5) * (1 + np.dot(np.dot(np.transpose(W), Sigma_inv), W) / (N - 1))**(- N / 2))
            s += -2 * np.log((1 + np.dot(np.dot(np.transpose(W), Sigma_inv), W) / (N - 1))**(- N / 2))
        
        chi2 = s / (n_simulations)
        return(np.asscalar(chi2))
    
    def likelihood_chi2_bm(self, Y_model, Noise_model, Y_observation, Noise_observation):
        s = 0
        
        n_simulations = np.shape(Y_observation)[0]
        
        for i in range(n_simulations):
            Sigma_model = Noise_model[i]
            Sigma_observation = Noise_observation[i]
            
            Sigma = Sigma_model + Sigma_observation
            # Sigma = 2 * Sigma_observation
            Sigma_inv = np.linalg.inv(Sigma)
            
            U = Y_model[i]
            V = Y_observation[i]
            W = U - V
            
            s += np.dot(np.dot(np.transpose(W), Sigma_inv), W)
            #s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
        
        chi2 = s / (n_simulations)
        return(np.asscalar(chi2))