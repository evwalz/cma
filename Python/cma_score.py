import numpy as np
from scipy.stats import rankdata

def compute_cma(y, x):
    y_rank = rankdata(y, method='average')
    x_rank = rankdata(x, method='average') 
    N = len(y)
    #mean_rank = (N+1)/2
    var = np.sum((y_rank - np.mean(y_rank))**2)*(1/(N-1))
    return (np.cov(y_rank,x_rank)[0][1]/var+1)/2