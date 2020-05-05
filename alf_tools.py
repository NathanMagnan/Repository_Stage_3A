import numpy as np
import treecorr as treecorr
import mistree as mist

def get_data(count, alpha, beta, gamma, t0, ts, box_size, mode):
    X, Y, Z = mist.get_adjusted_levy_flight(size = count, alpha = alpha, beta = beta, gamma = gamma, t_0 = t0, t_s = ts, box_size = box_size, mode = mode)
    halos_unbiased = {'haloCM' : []}
    for i in range(count):
        halos_unbiased['haloCM'].append(np.asarray([X[i], Y[i], Z[i]]))
    
    return(np.asarray(halos_unbiased['haloCM']))

def get_2PCF(bin_min, bin_max, n_bin, box_size, count, alpha, beta, gamma, t0, ts, mode):
    Bins = np.logspace(np.log10(bin_min), np.log10(bin_max), n_bin)
    
    List_xi = []
    for i in range(50):
        X, Y, Z = mist.get_adjusted_levy_flight(size = count, alpha = alpha, beta = beta, gamma = gamma, t_0 = t0, t_s = ts, box_size = box_size, mode = mode)
        data = treecorr.Catalog(x = X, y = Y, z = Z)
        dd = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin)
        dd.process(data)
        
        pos_uniform = np.random.rand(len(X), 3)
        X_uniform = pos_uniform[:-1, 0] * box_size
        Y_uniform = pos_uniform[:-1, 1] * box_size
        Z_uniform = pos_uniform[:-1, 2] * box_size
        uniform_distribution = treecorr.Catalog(x = X_uniform, y = Y_uniform, z = Z_uniform)
        uu = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin)
        uu.process(uniform_distribution)
    
        xi, varxi = dd.calculateXi(uu) #The 2PCF compare the data distribution to an uniform distribution
        List_xi.append(xi)
    
    Mean_xi = [0 for r in Bins]
    Std_xi = [0 for r in Bins]
    for i in range(50):
        for k in range(len(Bins)):
            mean_old = Mean_xi[k]
            std_old = Std_xi[k]
            x_new = List_xi[i][k]
            mean_new = (i * mean_old + x_new) / (i + 1)
            std_new = np.sqrt((i * (std_old**2 + mean_old**2) + x_new**2) / (i + 1) - mean_new**2)
            Mean_xi[k] = mean_new
            Std_xi[k] = std_new
    Mean_xi = np.asarray(Mean_xi)
    Std_xi = np.asarray(Std_xi)
    
    return(Bins, Mean_xi, Std_xi)

def get_MST_histogram(mode_MST, MST = None, count = None, alpha = None, beta = None, gamma = None, t0 = None, ts = None, box_size = None, mode = None):
    histogram = mist.HistMST()
    histogram.setup(usenorm = False, uselog = True)
    if (mode_MST == 'SingleMST'):
        d, l, b, s, l_index, b_index = MST.get_stats(include_index = True)
        return(histogram.get_hist(d, l, b, s))
    else:
        histogram.start_group()
        
        for i in range(5):
            X, Y, Z = mist.get_adjusted_levy_flight(size = count, alpha = alpha, beta = beta, gamma = gamma, t_0 = t0, t_s = ts, box_size = box_size, mode = mode)
            
            MST = mist.GetMST(x = X, y = Y, z = Z)
            d, l, b, s, l_index, b_index = MST.get_stats(include_index=True)
            _hist = histogram.get_hist(d, l, b, s)
        
        return(histogram.end_group())