## Imports
import numpy as np
import scipy.spatial as spatial
import GPy as GPy

## GP Class
class GP():
    
    def __init__(self, X_data, Y_data, kernel_type = None, Noise_std = None):
        """ To Be Tested"""
        self.x_dim =  np.shape(X_data)[1] # int
        self.y_dim =  np.shape(Y_data)[1] # int
        self.n_data_points = np.shape(X_data)[0] # int
        if (np.shape(Y_data)[0] != self.n_data_points):
            return ("Error : There is not as many X and Y data points : " + self.n_data_points + " vs. " + np.shape(Y_data)[0])
        
        self.X_data = X_data # np.array(n_data_points, x_dim)
        self.Y_data = Y_data # np.array(n_data_points, y_dim)
        self.Noise_std = Noise_std # np.array(y_dim, y_dim)
        
        self.kernel_type = kernel_type # str
        self.kernel = None # GPy.kern object
        self.initialise_kernel() # intialises kernel_parameters and kernel
        
        self.model = None # GPy.model object
        self.initialise_model() # initialises the GP regression model
        
        self.Y_prediction_on_test_points = None
        self.Cov_on_test_points = None
    
    def initialise_kernel(self, kernel = None):
        """ To Be Tested"""
        """ Needs to be improved to account for the selection of observables """
        if (kernel is not None):
            self.kernel = kernel
        
        List_kernel_types = self.kernel_type.split('+')
        
        for type in List_kernel_types :
            if (type == 'RBF'):
                self.kernel += GPy.kern.RBF(self.x_dim, ARD = True) # By default, we assume anistropy
            elif (type == 'Matern 3/2'):
                self.kernel += GPy.kern.Matern32(self.x_dim, ARD = True)
            elif (type == 'Matern 5/2'):
                self.kernel += GPy.kern.Matern52(self.x_dim, ARD = True)
            elif (type == 'Exponential'):
                self.kernel += GPy.kern.Exponential(self.x_dim, ARD = True)
            elif (type == 'White'):
                self.kernel += GPy.kern.White(self.x_dim, ARD = True)
            elif (type == 'Linear'):
                self.kernel += GPy.kern.Linear(self.x_dim, ARD = True)
            else:
                return("Error : kernel type " + type + " was not understood.")
            
    def initialise_model(self):
        """ To Be Tested """
        if (self.kernel is None):
            self.initialise_kernel()
        
        self.model = GPy.models.GPRegression(self.X_data, self.Y_data, self.kernel)
        
        if (self.Noise_std is not None):
            if (np.shape(self.Noise_std) != (self.y_dim, self.y_dim)):
                return("Error : Noise_std does not have the right dimensions : " + np.shape(self.Noise_std) + " vs. " + (self.y_dim, self.y_dim))
            
            self.model.Gaussian_noise.variance = self.Noise_std
            self.model.Gaussian_noise.variance.fix()
        else:
            self.model.Gaussian_noise.variance = 0.01 * np.identity(n = self.y_dim) """ To Be Defined """
    
    def optimize_model(self):
        """ To Be Tested """
        if (self.model is None):
            self.initialise_model()
        
        self.model.optimize()
    
    def compute_performance_on_tests(X_test, Y_test, Noise_std_test = None, Observables_to_use = None): # we assume the model either has already been optimized, or hasn't been initialized
        """ To Be Tested """
        """ Needs to be improved to account for the selection of observables """
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        n_test_points = np.shape(X_test)[0]
        if (n_test_points != np.shape(Y_test)[0]):
            return("Error : There is not as many X and Y test points : " + n_test_points + " vs. " + np.shape(Y_test)[0])
        if (np.shape(X_test)[1] != self.x_dim):
            return("Error : X_test does not have the right dimension : " + self.x_dim + " vs. " + np.shape(X_test)[1])
        if (np.shape(Y_test)[1] != self.y_dim):
            return("Error : Y_test does not have the right dimension : " + self.y_dim + " vs. " + np.shape(Y_test)[1])
        if (np.shape(Noise_std_test) != (self.y_dim, self.y_dim)):
            return("Error : Noise_std_test does not have the right dimensions : " + np.shape(Noise_std_test) + " vs. " + (self.y_dim, self.y_dim))
        
        Y_prediction_test, Cov_test = self.model.predict(X_test, full_cov = True) # This might definitely not work, Cov_test will have some strange shape
        
        performance = chi_2(Y_test, Noise_std_test, Y_prediction_test, Cov_test)
        return(performance)
    
    def compute_prediction(X_new, Observables_to_use = None): # we assume the model either has already been optimized, or hasn't been initialized
        """ To Be Tested """
        """ Needs to be improved to account for the selection of observables """
        if (self.model is None):
            self.initialise_model()
            self.optimize_model()
        
        n_new_points = np.shape(X_new)[0]
        if (np.shape(X_new)[1] != self.x_dim):
            return("Error : X_new does not have the right dimension : " + self.x_dim + " vs. " + np.shape(self.X_new)[1])
        
        Y_prediction, Cov = self.model.predict(X_new, full_cov = True) # This might definitely not work, Cov might have some strange shape
        return(Y_prediction, Cov)
    
    def plot_prediction(X_new, Observables_to_use = None):
        return("Error : Plot_prediction has yet to be written")
        


## Tools
def chi_2(Y_model, Noise_model, Y_observation, Noise_observations):
    """ To Be Tested """
    result = 0
    
    for i  in range(np.shape(Y_model)[0]):
        u = Y_model[i]
        v = Y_observation[i]
        Cov = np.asarray(Noise_model) + np.asarray(Noise_observations)
        VI = np.linalg.inv(Cov)
        result += spatial.distance.mahalanobis(u, v, VI)
    
    return(result)