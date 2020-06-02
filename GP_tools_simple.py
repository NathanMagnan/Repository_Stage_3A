## Imports
import numpy as np
import scipy.spatial as spatial
import GPy as GPy

## GP Class
class GP():
    
    def __init__(self, X_data, Y_data, kernel, noise_data = None, n_points_per_simulation = 6):
        self.n_data_points = np.shape(X_data)[0] # int
        self.n_points_per_simulation = n_points_per_simulation # int
        self.x_dim =  np.shape(X_data)[1] # int
        
        self.X_data = X_data # np.array(n_data_points, x_dim)
        
        self.Y_data = Y_data # np.array(n_data_points, 1)
        if (np.shape(Y_data)[0] != self.n_data_points):
            return ("Error : There is not as many X and Y data points : " + self.n_data_points + " vs. " + np.shape(Y_data)[0])
        self.Y_data = np.reshape(Y_data, (self.n_data_points, 1))
        
        self.noise_data = noise_data # float
        
        self.kernel = kernel # GPy.kern object
        
        self.model = None # GPy.model object
        self.initialise_model() # initialises the GP regression model
            
    def initialise_model(self):
        self.model = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.noise_data is not None):
            self.model.Gaussian_noise.variance = self.noise_data
            self.model.Gaussian_noise.variance.fix()
        else:
            self.model.Gaussian_noise.variance = 0
            self.model.Gaussian_noise.variance.fix()
    
    def optimize_model(self, optimizer = 'bfgs'):
        if (self.model is None):
            self.initialise_model()
        
        self.model.optimize(optimizer)
    
    def compute_performance_on_tests(self, X_test, Y_test, noise_test = None): # we assume the model either has already been optimized, or hasn't been initialized
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        n_test_points = np.shape(X_test)[0]
        
        if (np.shape(X_test)[1] != self.x_dim):
            return("Error : X_test does not have the right dimension : " + self.x_dim + " vs. " + np.shape(X_test)[1])
        
        if (np.shape(Y_test)[0] != n_test_points):
            return("Error : There is not as many X and Y test points : " + n_test_points + " vs. " + np.shape(Y_test)[0])
        
        Y_predicted_test, Cov_test = self.model.predict(X_test, full_cov = True)
        
        #performance = self.chi_2(Y_test, noise_test, Y_predicted_test, Cov_test)
        performance = self.RMS(Y_test, Y_predicted_test)
        return(performance)
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized, or hasn't been initialized
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_points = np.shape(X_new)[0]
        
        for i in range(n_new_points): # construct a full X matrix from the cosmological parameters
            h0, w0, ns, sigma8, omegaM = X_new[i]
            X = self.X_data[0:self.n_points_per_simulation, 5]
            X_new = np.reshape([[h0, w0, ns, sigma8, omegaM, x] for x in X], (self.n_points_per_simulation, 6))
        
        Y_predicted, Cov = self.model.predict(X_new, full_cov = True)
        return(Y_predicted, Cov)
    
    def plot_prediction(self, X_new):
        return("Error : Plot_prediction has yet to be written")
        


## Tools
    def chi_2(self, Y_model, noise_model, Y_observation, Noise_observations):
        chi2 = 0
        
        n_points_per_simulation = self.n_points_per_simulation
        n_simulations = np.shape(Y_model)[0] // n_points_per_simulation
        
        for i  in range(n_simulations): # we compute a chi2 for each simu then we sum the chi2s
            u = Y_model[n_points_per_simulation * i : n_points_per_simulation * (i + 1)]
            v = Y_observation[n_points_per_simulation * i : n_points_per_simulation * (i + 1)]
            Cov = np.identity(n_points_per_simulation) * noise_model + np.asarray(Noise_observations[n_points_per_simulation * i : n_points_per_simulation * (i + 1), n_points_per_simulation * i : n_points_per_simulation * (i + 1)])
            
            VI = np.linalg.inv(Cov)
            chi2 += spatial.distance.mahalanobis(u, v, VI)**2
        
        chi2 = np.sqrt(chi2 / n_simulations)
        
        return(chi2)
    
    def RMS(self, Y_model, Y_observation): # Root Mean Square Error
        s = 0
        
        n_points_per_simulation = self.n_points_per_simulation
        n_simulations = np.shape(Y_model)[0] // n_points_per_simulation
        
        for i  in range(n_simulations):
            u = np.asarray(Y_model[n_points_per_simulation * i : n_points_per_simulation * (i + 1)])
            v = np.asarray(Y_observation[n_points_per_simulation * i : n_points_per_simulation * (i + 1)])
            u = np.reshape(u, (n_points_per_simulation, 1))
            v = np.reshape(v, (n_points_per_simulation, 1))
            
            s += np.sum((u - v)**2)
        
        rms = np.sqrt(s / n_simulations)
        
        return(rms)