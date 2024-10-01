from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.stats import chi2
from scipy.stats import norm
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
from scipy.stats import rankdata
from scipy.special import comb
import numpy as np
from scipy.stats import norm
import pandas as pd
from numba import njit, prange


# Helper function to calculate combinations (n choose k)
@njit
def comb(n, k):
    if n < k:
        return 0
    if k == 0:
        return 1
    num = 1
    denom = 1
    for i in range(k):
        num *= n - i
        denom *= i + 1
    return num // denom  # Using integer division to avoid float results

# JIT compilation with numba for kernel_p
@njit(parallel=True)
def kernel_p(x_rank, y_rank, rho):
    N = len(y_rank)
    
    # Precompute G_x and G_y
    G_x = (x_rank - 0.5) / N
    G_y = (y_rank - 0.5) / N
    
    # Explicitly reshape arrays to avoid issues with `None`
    s_x = np.sign(x_rank.reshape(N, 1) - x_rank)  # Reshaping instead of x_rank[:, None]
    s_y = np.sign(y_rank.reshape(N, 1) - y_rank)  # Reshaping instead of y_rank[:, None]

    # Vectorized computation of mean_product with numba
    all_exp = np.zeros((N, N))
    for i in prange(N):
        for j in prange(N):
            all_exp[i, j] = np.mean(s_x[i, :] * s_y[j, :]) + 2 * G_x[i] + 2 * G_y[j] - 1
    
    # Manually compute the mean along axis=0 and axis=1
    g_1 = np.zeros(N)
    g_2 = np.zeros(N)
    
    for i in prange(N):
        g_1[i] = np.sum(all_exp[:, i]) / N  # Mean along axis=0
        g_2[i] = np.sum(all_exp[i, :]) / N  # Mean along axis=1
    
    g_1 /= 4
    g_2 /= 4
    
    # Calculate k_p using vectorized operations
    k_p = 4 * (g_1 + g_2 + G_x * G_y - G_y - G_x) + 1 - rho
    return k_p

# Manual implementation of unique and counts since numba doesn't support np.unique with return_counts
@njit
def unique_with_counts(arr):
    sorted_indices = np.argsort(arr)
    sorted_arr = arr[sorted_indices]
    
    unique_vals = []
    counts = []
    
    count = 1
    for i in range(1, len(arr)):
        if sorted_arr[i] == sorted_arr[i - 1]:
            count += 1
        else:
            unique_vals.append(sorted_arr[i - 1])
            counts.append(count)
            count = 1
    
    # Add the last element
    unique_vals.append(sorted_arr[-1])
    counts.append(count)
    
    return np.array(unique_vals), np.array(counts)

# Numba-optimized zeta function
@njit
def zeta_fun(y):
    N = len(y)
    if N < 3:
        return 0
    
    unique, counts = unique_with_counts(y)
    triplets_count = 0
    for count in counts:
        if count >= 3:
            triplets_count += comb(count, 3)

    bin_N_3 = 6 / (N * (N - 1) * (N - 2))
    return bin_N_3 * triplets_count

# Optimized probability computation using numba
@njit
def prob_y(y):
    unique, counts = unique_with_counts(y)
    probabilities = counts / len(y)
    return probabilities[np.searchsorted(unique, y)]

# Numba-optimized function to compute rho and cma values
@njit
def comp_rho_cma(y_rank, x_rank):
    N = len(y_rank)
    mean_rank = (N + 1) / 2
    var_y =  np.sum((y_rank - np.mean(y_rank))**2)*(1/(N-1))
    rho_val = (12 / (N ** 2)) * (1 / (N - 1)) * np.sum((x_rank - mean_rank) * (y_rank - mean_rank))
    cma_val = (np.cov(y_rank, x_rank)[0, 1] / var_y + 1) / 2
    return rho_val, cma_val

# Sigma function optimized with JIT compilation
def Sigma_fast(y_rank, xarray_ranks):
    N = len(y_rank)
    k = xarray_ranks.shape[0]

    zeta_3Y = zeta_fun(y_rank)
    k_zeta = prob_y(y_rank) ** 2 - zeta_3Y
    sigma_zeta = 9 * np.mean(k_zeta ** 2)

    rhos = np.zeros(k)
    cmas = np.zeros(k)
    
    # Precompute rho and CMA for all ranks
    for j in prange(k):
        rhos[j], cmas[j] = comp_rho_cma(y_rank, xarray_ranks[j, :])

    factor = 1 / ((1 - zeta_3Y) ** 2)
    phalf = np.zeros(k)
    S = np.zeros((k, k))

    for j in prange(k):
        k_p = kernel_p(xarray_ranks[j, :], y_rank, rhos[j])
        sigma_rho = 9 * np.mean(k_p ** 2)
        sigma_pz = 9 * np.mean(k_p * k_zeta)
        
        var = factor * (
            sigma_rho + (2 * rhos[j] * sigma_pz) / (1 - zeta_3Y) +
            (rhos[j] ** 2 * sigma_zeta) / ((1 - zeta_3Y) ** 2)
        )
        S[j, j] = var / (4 * N)
        phalf[j] = 1 - norm.cdf((cmas[j] - 0.5) / np.sqrt(var / (4 * N)))

        # Calculate off-diagonal elements
        for i in prange(j + 1, k):
            k_p2 = kernel_p(xarray_ranks[i, :], y_rank, rhos[i])
            sigma_rho2 = 9 * np.mean(k_p * k_p2)
            sigma_pz2 = 9 * np.mean(k_p2 * k_zeta)
            
            var = factor * (
                sigma_rho2 + (rhos[j] * sigma_pz) / (1 - zeta_3Y) +
                (rhos[i] * sigma_pz2) / (1 - zeta_3Y) +
                (rhos[j] * rhos[i] * sigma_zeta) / ((1 - zeta_3Y) ** 2)
            )
            S[j, i] = S[i, j] = var / (4 * N)

    # Return DataFrame with results
    cmas_pd = pd.DataFrame({
        'CMA': cmas,
        'SD': np.sqrt(np.diag(S)),
        'P(H0: CMA=0.5)': phalf
    })

    return cmas, S, cmas_pd




def one_dim_test(y_rank, x_rank):
    N = len(y_rank)


    zeta_3Y = zeta_fun(y_rank)
    k_zeta = prob_y(y_rank)**2 - zeta_3Y
    sigma_zeta = 9*np.mean(k_zeta**2)

    rho, cmas = comp_rho_cma(y_rank, x_rank)

    factor = 1 / ((1-zeta_3Y)**2)


    k_p =  kernel_p(x_rank, y_rank, rho)
    sigma_rho = 9*np.mean(k_p**2)
    sigma_pz = 9*np.mean((k_p * k_zeta)) 
    var = factor*(sigma_rho + (2*rho*sigma_pz)/(1-zeta_3Y) + (rho**2*sigma_zeta)/((1-zeta_3Y)**2))
    sd_2 = var/(4*N)
    phalf = 1 - norm.cdf((cmas - 0.5) / np.sqrt(var/(4*N)))

    return cmas, np.sqrt(sd_2), phalf


class test_multiple(object):

    def __init__(self, cmas, differences, covariance, global_p, global_z):
        self.cmas = cmas
        self.differences = differences
        self.covariance = covariance
        self.global_z = global_z
        self.global_p = global_p


    def print(self):
        print('CMA test: \n', self.cmas)
        print('\n')
        print('Pairwise test: \n', self.differences)
        print('\n')
        print('Covariance: \n', self.covariance)
        print('\n')
        print('Global z-value: ' , self.global_z)
        print('Global p-value: ' ,self.global_p)

class test_one(object):

    def __init__(self, cmas, sd, p):
        self.cmas = cmas
        self.sd = sd
        self.p = p

    def print(self):
        output_pd = pd.DataFrame({'CMA': [self.cmas], 'SD': [self.sd], 'P(H0: CMA = 0.5)': self.p})
        print('CMA test: \n', output_pd)


def calc_pvalue_chi_our(aucs, S):
    nauc = len(aucs)

    # Initialize L matrix with zeros
    L = np.zeros((nauc*(nauc-1)//2, nauc))

    newa = 0
    for i in range(nauc-1):
        newl = nauc - (i + 1)
    
        # Assign 1 to the first part of the slice
        L[newa:(newa+newl), i] = np.ones(newl)
    
        # Assign diagonal -1 values to the next part of the slice
        L[newa:(newa+newl), (i+1):(i+1+newl)] = -np.eye(newl)
    
        newa += newl


    aucdiff = L @ aucs
    L_S_Lt = L @ S @ L.T 
    # use R function from rms library matinv
    numpy2ri.activate()
    rms = importr('rms')
    # Convert the numpy array to an R object
    r_matrix = ro.r['as.matrix'](L_S_Lt)
    
    L_S_Lt_inv = np.array(rms.matinv(r_matrix))
    z = aucdiff.T @ L_S_Lt_inv @ aucdiff
    
    # Compute degrees of freedom using rank of the matrix
    _, R = np.linalg.qr(L_S_Lt)  # QR decomposition to get rank
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)  # Count non-zero diagonal elements

    # Calculate p-value using chi-squared distribution
    p = chi2.sf(z, df=rank) 
    return z, p, aucdiff


def pairwise_testing_our(nauc, S, aucdiff, conf_level):
    cor_auc = np.zeros((nauc * (nauc - 1)) // 2)
    ci = np.zeros(((nauc * (nauc - 1)) // 2, 2))
    pairp = np.zeros((nauc * (nauc - 1)) // 2)
    rows = []
    ctr = 0
    quantil = norm.ppf(1 - (1 - conf_level) / 2)
    numpy2ri.activate()
    rms = importr('rms')
    # Loop through pairs of AUCs
    for i in range(nauc - 1):
        for j in range(i + 1, nauc):
            cor_auc[ctr] = S[i, j] / np.sqrt(S[i, i] * S[j, j])
        
            # Compute LSL
            LSL = np.dot(np.dot(np.array([1, -1]), S[[j, i], :][:, [j, i]]), np.array([1, -1]))
    
            # Compute tmpz and pairp
            tmpz = aucdiff[ctr] / np.sqrt(LSL)
            pairp[ctr] = chi2.sf(aucdiff[ctr]**2 / LSL, df=1)
        
            # Compute confidence interval
            ci[ctr, 0] = aucdiff[ctr] - quantil * np.sqrt(LSL)
            ci[ctr, 1] = aucdiff[ctr] + quantil * np.sqrt(LSL)
        
            # Track row names (i vs j)
            rows.append(f"{i+1} vs. {j+1}")
            ctr += 1
        
    return pd.DataFrame({'Test': rows, 'CMA diff': aucdiff, 'CI(lower)': ci[:, 0] ,'CI(upper)': ci[:, 1], 'p.value': pairp,'correlation':cor_auc})



def cma_test_fast(y, x ,conf_level = 0.95):
    ##################
    # checking N > 3 #
    ##################
    
    # single prediction for y:
    if x.ndim == 1:
        # computation performed on ranks:
        y_ranks = rankdata(y, method='average')
        x_ranks = rankdata(x, method='average')
        
        # compute score, variance estimation and p-value
        cmas, sd, p = one_dim_test(y_ranks, x_ranks)
        
        return test_one(cmas = cmas, sd = sd, p = p)
    
    # multiple predictors for y:
    else:
        # computation performed on ranks:
        y_ranks = rankdata(y, method='average')
        xarray_ranks = np.apply_along_axis(rankdata, axis=1, arr=x, method='average')

        # single value testing and variance estimation: 
        cmas, S , cma_pd = Sigma_fast(y_ranks, xarray_ranks)

        # global testing:
        z, p, cmadiff = calc_pvalue_chi_our(cmas, S)

        # pairwise testing:
        diff_pd = pairwise_testing_our(len(cmas), S, cmadiff, conf_level)
    
        return test_multiple(cmas = cma_pd, differences = diff_pd, covariance = S, global_z = z, global_p = p)