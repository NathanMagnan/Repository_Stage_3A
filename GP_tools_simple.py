## Imports
import numpy as np
import scipy.spatial as spatial
import scipy.linalg as linalg
import GPy as GPy

## GP Class
class GP():
    
    def __init__(self, X, Y, n_points_per_simu = 5, Noise = None):
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
    
    def make_kernel(self):
        self.kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
    
    def make_model(self):
        gp = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.Noise is not None):
            gp = GPy.models.GPHeteroscedasticRegression(self.X_data, self.Y_data, self.kernel)
            gp['.*het_Gauss.variance'] = abs(self.Noise)
            gp.het_Gauss.variance.fix()
        
        self.model = gp
    
    def change_kernels(self, new_kernel):
        self.kernel = new_kernel
        self.make_model()
    
    def change_model_to_heteroscedatic(self, Noise = None):
        self.Noise = Noise # None or (1, n_points) array
        
        gp = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.Noise is not None):
            gp = GPy.models.GPHeteroscedasticRegression(self.X_data, self.Y_data, self.kernel)
            gp['.*het_Gauss.variance'] = abs(self.Noise)
            gp.het_Gauss.variance.fix()
        
        self.model = gp
            
    def optimize_model(self, optimizer = 'lbfgsb'):
        self.model.optimize(optimizer = optimizer)
    
    def print_model(self):
        gp = self.model
        
        print("GP model :")
        print(gp.rbf.variance)
        print(gp.rbf.lengthscale)
        if (self.Noise is not None):
            #print(gp.het_Gauss.variance)
            print()
        else:
            print(gp.Gaussian_noise.variance)
        print("Obj : " + str(gp.objective_function()))
        print("\n")
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_cosmologies = np.shape(X_new)[0]
        
        X_predicted, Y_predicted, Cov = [], [], []
        for i in range(n_new_cosmologies):
            h0, w0, ns, sigma8, omegaM = X_new[i]
            
            X = self.X_data[0 : self.n_points_per_simu, 5]
            X_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x] for x in X], (self.n_points_per_simu, 6))
            Y_new, Cov_new = self.model.predict(X_new, full_cov = True, Y_metadata = {'output_index' : np.array([0])})
            X_predicted.append(X_new)
            Y_predicted.append(Y_new)
            Cov.append(Cov_new)
        return(X_predicted, Y_predicted, Cov)
    
    def test_rms(self, X_test, Y_test, Noise_test = None):
        s = 0
        for i in range(np.shape(X_test)[0]):
            X_new = np.reshape(X_test[i], (1, 6))
            y_predicted = (self.model.predict(X_new, full_cov = True, Y_metadata = {'output_index' : np.array([0])}))[0][0][0]
            y_expected = Y_test[i][0]
            s += (y_predicted - y_expected)**2
        
        ms = s / np.shape(X_test)[0]
        
        rms = np.sqrt(ms)
        return(rms)
    
    def test_chi2(self, X_test, Y_test, Noise_test = None):
        if (Noise_test is not None):
            Y_std_test = Noise_test
        
        s = 0            
        for i in range(np.shape(X_test)[0]):
            X_new = np.reshape(X_test[i], (1, 6))
            y_predicted, c = (self.model.predict(X_new, full_cov = True, Y_metadata = {'output_index' : np.array([0])}))
            y_expected, noise = Y_test[i][0], Y_std_test[i][0]
            print(y_predicted, y_expected, noise, np.sqrt(c[0][0]))
            s += (y_predicted[0][0] - y_expected)**2 / (c[0][0] + noise**2)
        
        ms = s / np.shape(X_test)[0]
        
        chi_2 = ms
        return(chi_2)
    
    def likelihood_chi2(self, Y_model, Noise_model, Y_observation, Noise_observation):
        s = 0
        
        n_simulations = int(np.shape(Y_observation)[0] / self.n_points_per_simu)
        
        for i in range(n_simulations):
            Sigma_observation = Noise_observation[i]
            if (Sigma_observation is None):
                Diag = [self.model.Gaussian_noise.variance for j in range(self.n_points_per_simu)]
                Sigma_observation = np.diag(Diag)
            
            Sigma_model = Noise_model[i]
            Sigma = Sigma_model + Sigma_observation
            Sigma_inv = np.linalg.inv(Sigma)
            
            U = Y_observation[i * self.n_points_per_simu : (i + 1) * self.n_points_per_simu]
            V = Y_model[i * self.n_points_per_simu : (i + 1) * self.n_points_per_simu]
            
            s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
        
        chi2 = s / (n_simulations)
        return(chi2)
    
    def likelihood_ms(self, Y_model, Y_observation, Noise_model = None, Noise_observation = None):
        s = 0
        
        n_simulations = int(np.shape(Y_observation)[0] / self.n_points_per_simu)
        
        for i in range(n_simulations):
            Sigma_inv = np.identity(self.n_points_per_simu)
            
            U = Y_observation[i * self.n_points_per_simu : (i + 1) * self.n_points_per_simu]
            V = Y_model[i * self.n_points_per_simu : (i + 1) * self.n_points_per_simu]
            
            s += spatial.distance.mahalanobis(U, V, Sigma_inv)**2
        
        ms = s / (n_simulations)
        return(ms)
    
    def plot_prediction(self, X_new):
        return("Error : Plot_prediction has yet to be written")