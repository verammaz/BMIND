import numpy as np
import pandas as pd
import cmdstanpy
from scipy.optimize import nnls



def log2_cpm(bulk):
    """
    Apply log2 counts per million (CPM) transformation if the maximum value in bulk is greater than 50.
    """
    if np.max(bulk) > 50:
        bulk = np.log2((bulk / np.sum(bulk, axis=0) * 1e6) + 1)
    return bulk


def estimate_frac_nnls(bulk, signature):
    """
    Estimate cell type fractions using Non-Negative Least Squares (NNLS).
    
    bulk: (genes x samples) expression matrix --> log2_cpm transformed
    signature: (genes x cell types) signature matrix  

    Returns:
    frac: (samples x cell types) estimated cell type proportions
    """

    bulk = log2_cpm(bulk)

    if np.max(signature) > 50:
        signature = np.log2(signature + 1) # log2 transformation

    frac = np.zeros((bulk.shape[1], signature.shape[1]))  # initialize fraction matrix
    
    for i in range(bulk.shape[1]):  # loop over samples
        frac[i], _ = nnls(signature, bulk[:, i])  # solve NNLS for each sample

    return frac

def run_stan_model(bulk, frac, profile, covariance, nu=50):
    """
    Runs the Stan model to estimate cell-type-specific gene expression.
    
    bulk: (G x S) bulk expression matrix
    frac: (S x K) cell type fractions
    profile: (G x K) prior mean expression matrix
    covariance: (G x K x K) prior covariance array
    nu: hyperparameter for covariance strength
    
    Returns:
    A_estimated: (G x K) estimated cell-type-specific expression
    """

    data = {
    'G': bulk.shape[0],   # Number of genes
    'S': bulk.shape[1], # Number of samples
    'K': frac.shape[1], # Number of cell types
    'W': frac,    # Cell type fractions (S x K matrix)
    'X': bulk,    # Bulk expression data (G x S matrix)
    'a_hat': profile,  # Prior mean CTS expression (G x K matrix)
    'S_hat': covariance,  # Prior covariance matrix for each gene (G x K matrix)
    }

    return 

