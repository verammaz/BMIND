// bMIND Stan Model
//
// This model implements the Bayesian Mixed-effects model for 
// estimating cell type-specific (CTS) gene expression from bulk
// RNA-seq data, using prior information from single-cell RNA-seq.
//
// The model is based on:
// Wang et al. (2021) - Bayesian estimation of cell type–specific gene
// expression with prior derived from single-cell data

data {
  // Dimensions
  int<lower=1> N;                   // Number of samples
  int<lower=1> G;                   // Number of genes
  int<lower=1> K;                   // Number of cell types
  int<lower=1> T;                   // Number of measures per sample (can be 1)
  int<lower=0> P1;                  // Number of covariates affecting bulk expression
  int<lower=0> P2;                  // Number of covariates affecting CTS expression
  
  // Data matrices
  matrix[T, K] W[N];                // Cell type fraction matrix for each sample
  vector[T] X[N, G];                // Bulk expression data for each sample, gene
  matrix[N, P1] C1;                 // Covariates affecting bulk expression
  matrix[N, P2] C2;                 // Covariates affecting CTS expression
  
  // Prior information from single-cell data
  vector[K] prior_mean[G];          // Prior means for each gene across cell types
  matrix[K, K] prior_cov[G];        // Prior covariance matrices
}

parameters {
  // Cell type-specific expression for each sample, gene
  vector[K] a[N, G];                
  
  // Covariate effects
  vector[P1] beta[G];               // Effects of covariates on bulk expression
  matrix[K, P2] B[G];               // Effects of covariates on CTS expression
  
  // Error and covariance parameters
  real<lower=0> sigma[G];           // Error standard deviation for each gene
  cov_matrix[K] Sigma[G];           // Covariance matrix of CTS expression for each gene
}

model {
  // Priors
  for (g in 1:G) {
    // Prior for covariance matrix - from single-cell data
    Sigma[g] ~ inv_wishart(50, prior_cov[g]);
    
    // Prior for covariate effects
    beta[g] ~ normal(0, 1);
    for (k in 1:K) {
      B[g, k] ~ normal(0, 1);
    }
    
    // Prior for error variance - non-informative
    sigma[g] ~ cauchy(0, 5);
    
    // Prior for CTS expression - from single-cell data
    for (i in 1:N) {
      a[i, g] ~ multi_normal(prior_mean[g], Sigma[g]);
    }
  }
  
  // Likelihood
  for (i in 1:N) {
    for (g in 1:G) {
      for (t in 1:T) {
        real mu;
        
        // Base model: mu = W * a + C1 * beta
        mu = dot_product(W[i, t], a[i, g]) + dot_product(C1[i], beta[g]);
        
        // Add effect of covariates on CTS expression
        for (k in 1:K) {
          mu = mu + W[i, t, k] * dot_product(B[g, k], C2[i]);
        }
        
        // Likelihood of observed expression
        X[i, g, t] ~ normal(mu, sigma[g]);
      }
    }
  }
}

generated quantities {
  // Predicted bulk expression for model checking
  vector[T] X_pred[N, G];
  
  // Log-likelihood for model comparison
  matrix[N, G] log_lik;
  
  for (i in 1:N) {
    for (g in 1:G) {
      for (t in 1:T) {
        real mu;
        
        // Calculate predicted expression
        mu = dot_product(W[i, t], a[i, g]) + dot_product(C1[i], beta[g]);
        
        for (k in 1:K) {
          mu = mu + W[i, t, k] * dot_product(B[g, k], C2[i]);
        }
        
        X_pred[i, g, t] = mu;
        
        // Calculate log-likelihood (for first measure only to simplify)
        if (t == 1) {
          log_lik[i, g] = normal_lpdf(X[i, g, t] | mu, sigma[g]);
        }
      }
    }
  }
}