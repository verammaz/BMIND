data {

    int<lower=1> G; // Number of genes
    int<lower=1> S; // Number of samples
    int<lower=1> K; // Number of cell types

    matrix[S, K] W; // Cell type fractions (preestimated)
    matrix[G, S] X; // Bulk RNA-seq data

    matrix[G, K] a_hat; // Prior mean CTS expression from scRNA-seq
    cov_matrix[K] S_hat[G]; // Prior covariance matrix from scRNA-seq
    
}

parameters {
    matrix[G, K] A[S];  // Cell-type-specific expression per sample
    cov_matrix[K] S[G];  // Covariance matrix for CTS expression
    real<lower=0> sigma[G]; // Gene-specific variance
}

model {
    for (g in 1:G) {
        S[g] ~ inv_wishart(50, S_hat[g]); // Prior on covariance
        sigma[g] ~ inv_wishart(1, 0); // Non-informative prior on variance

        for (s in 1:S) {
            A[s][g] ~ multi_normal(a_hat[g], 0.5 * identity_matrix(K)); // Prior for CTS expression

            X[g, s] ~ normal(W[s] * A[s][g]', sigma[g]); // Bulk expression model
        }
    }
}
