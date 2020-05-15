## Imports
import numpy as np
import scipy.spatial as spatial
import GPy as GPy

## GP Class
class GP():
    
    def __init__(self, X_data, Y_data, kernel, noise_data = None):
        """ To Be Tested"""
        self.n_data_points = np.shape(X_data)[0] # int
        self.x_dim =  np.shape(X_data)[1] # int
        
        self.X_data = X_data # np.array(n_data_points, x_dim)
        
        self.Y_data = Y_data # np.array(n_data_points, 1)
        if (np.shape(Y_data)[0] != self.n_data_points):
            return ("Error : There is not as many X and Y data points : " + self.n_data_points + " vs. " + np.shape(Y_data)[0])
        self.Y_data = np.reshape(Y_data, (self.n_data_points, 1))
        
        # self.Noise_data = Noise_data # np.array(n_data_points, n_data_points)
        # if (self.Noise_data is not None):
            # if (np.shape(self.Noise_data) != (self.n_data_points, self.n_data_points)):
                # return("Error : Noise_std does not have the right dimensions : " + np.shape(self.Noise_data) + " vs. " + (self.n_data_points, self.n_data_points))
        self.noise_data = noise_data # float
        
        
        self.kernel = kernel # GPy.kern object
        
        self.model = None # GPy.model object
        self.initialise_model() # initialises the GP regression model
            
    def initialise_model(self):
        """ To Be Tested """
        self.model = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        #if (self.Noise_data is not None):
        if (self.noise_data is not None):
            # self.model.Gaussian_noise.variance = self.Noise_data # GPy does not accept this
            self.model.Gaussian_noise.variance = self.noise_data
            self.model.Gaussian_noise.variance.fix()
        else:
            self.model.Gaussian_noise.variance = 0
            self.model.Gaussian_noise.variance.fix()
    
    def optimize_model(self):
        """ To Be Tested """
        if (self.model is None):
            self.initialise_model()
        
        self.model.optimize()
    
    def compute_performance_on_tests(self, X_test, Y_test, noise_test = None): # we assume the model either has already been optimized, or hasn't been initialized
        """ To Be Tested """
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        n_test_points = np.shape(X_test)[0]
        
        if (np.shape(X_test)[1] != self.x_dim):
            return("Error : X_test does not have the right dimension : " + self.x_dim + " vs. " + np.shape(X_test)[1])
        
        if (np.shape(Y_test)[0] != n_test_points):
            return("Error : There is not as many X and Y test points : " + n_test_points + " vs. " + np.shape(Y_test)[0])
        
        # if (np.shape(Noise_test) != (n_test_points, n_test_points)):
            # return("Error : Noise_test does not have the right dimensions : " + np.shape(Noise_test) + " vs. " + (n_test_points, n_test_points))
        
        Y_predicted_test, Cov_test = self.model.predict(X_test, full_cov = True)
        
        performance = chi_2(Y_test, noise_test, Y_predicted_test, Cov_test)
        return(performance)
    
    def compute_prediction(self, X_new): # we assume the model either has already been optimized, or hasn't been initialized
        """ To Be Tested """
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        if (np.shape(X_new)[1] != 5): # we predict the MST for a set of cosmological parameters
            return("Error : X_new does not have the right dimension : " + str(5) + " vs. " + np.shape(X_new)[1])
        
        n_new_points = np.shape(X_new)[0]
        
        for i in range(n_new_points): # construct a full X matrix from the cosmological parameters
            h0, w0, ns, sigma8, omegaM = X_new[i]
            X_d = self.X_data[0:6, 5]
            X_l = self.X_data[6:16, 5]
            X_b = self.X_data[16:26, 5]
            X_s = self.X_data[26:36, 5]
            X_new_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d, 0] for x_d in X_d], (6, 7))
            X_new_l = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l, 1] for x_l in X_l], (10, 7))
            X_new_b = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b, 2] for x_b in X_b], (10, 7))
            X_new_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s, 3] for x_s in X_s], (10, 7))
            X_new_full = np.concatenate((X_new_d, X_new_l, X_new_b, X_new_s), 0)
        
        Y_predicted, Cov = self.model.predict(X_new_full, full_cov = True)
        return(Y_predicted, Cov)
    
    def plot_prediction(self, X_new):
        return("Error : Plot_prediction has yet to be written")
        


## Tools
def chi_2(Y_model, noise_model, Y_observation, Noise_observations):
    """ To Be Tested """
    chi2 = 0
    
    for i  in range(np.shape(Y_model)[0] // 36): # we compute a chi2 for each simu then we sum the chi2s
        u = Y_model[36 * i : 36 * (i + 1)]
        v = Y_observation[36 * i : 36 * (i + 1)]
        Cov = np.identity(36) * noise_model + np.asarray(Noise_observations[36 * i : 36 * (i + 1), 36 * i : 36 * (i + 1)])
        
        VI = np.linalg.inv(Cov)
        chi2 += spatial.distance.mahalanobis(u, v, VI)**2
    
    return(chi2)