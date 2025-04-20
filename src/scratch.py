import pandas as pd
import numpy as np
import pystan
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statsmodels.graphics.tsaplots import plot_acf
import corner  # For posterior distribution plots


from bMIND_func import bmind_model

def run_bmind_sequential(bulk_data, cell_fractions, prior_means, prior_covs, 
                         iter=2000, chains=4):
    """
    Run bMIND model on bulk RNA-seq data with sequential processing.
    """
    # Filter data
    genes = list(prior_means.keys())
    samples = bulk_data.columns.tolist()
    
    bulk_subset = bulk_data.loc[genes, samples]
    fractions_subset = cell_fractions.loc[samples]
    
    # Check dimensions
    N = len(samples)
    G = len(genes)
    K = fractions_subset.shape[1]
    
    print(f"Data dimensions: {N} samples, {G} genes, {K} cell types")
    
    # Default covariates
    C1 = pd.DataFrame(np.ones((N, 1)), index=samples, columns=['intercept'])
    C2 = pd.DataFrame(np.ones((N, 1)), index=samples, columns=['intercept'])
    
    P1 = C1.shape[1]
    P2 = C2.shape[1]
    
    # Create arrays of prior means and covariances
    prior_mean_array = np.zeros((G, K))
    prior_cov_array = np.zeros((G, K, K))
    
    for i, gene in enumerate(genes):
        prior_mean_array[i, :] = prior_means[gene]
        prior_cov_array[i, :, :] = prior_covs[gene]
    
    # Convert all data to numpy arrays
    stan_data = {
        'N': int(N),
        'G': int(G),
        'K': int(K),
        'P1': int(P1),
        'P2': int(P2),
        'W': fractions_subset.values.astype(np.float64),
        'X': bulk_subset.T.values.astype(np.float64),
        'C1': C1.values.astype(np.float64),
        'C2': C2.values.astype(np.float64),
        'prior_mean': prior_mean_array.astype(np.float64),
        'prior_cov': prior_cov_array.astype(np.float64)
    }
    
    # Compile Stan model
    sm = pystan.StanModel(model_code=bmind_model)
    
    # Run Stan with n_jobs=1 to avoid multiprocessing
    fit = sm.sampling(
        data=stan_data, 
        iter=iter, 
        chains=chains, 
        n_jobs=1  # This forces sequential processing
    )
    
    # Extract CTS expression estimates
    cts_expression = {}
    
    for g_idx, gene in enumerate(genes):
        gene_cts = pd.DataFrame(
            index=samples,
            columns=fractions_subset.columns,
            dtype=float         
        )
        
        for i, sample in enumerate(samples):
            for k, cell_type in enumerate(fractions_subset.columns):
                param_name = f'a[{i+1},{g_idx+1},{k+1}]'
                gene_cts.loc[sample, cell_type] = fit.extract(pars=[param_name])[param_name].mean()
        
        cts_expression[gene] = gene_cts
    
    return cts_expression, fit


def visualize_cts_expression(cts_expression, signature, cell_fractions, output_dir="results"):
    """
    Create visualizations of cell type-specific expression.
    
    Parameters:
    -----------
    cts_expression : dict
        Dictionary of DataFrames with cell type-specific expression estimates
    signature : pd.DataFrame
        Signature matrix used as prior
    cell_fractions : pd.DataFrame
        Cell type fractions
    output_dir : str
        Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Heatmaps of cell type-specific expression for each gene
    print("Creating heatmaps of cell type-specific expression...")
    
    for gene in list(cts_expression.keys()):  
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            cts_expression[gene], 
            cmap="viridis", 
            annot=True, 
            fmt=".2f",
            linewidths=0.5
        )
        plt.title(f"Cell Type-Specific Expression: {gene}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_{gene}.png", dpi=300)
        plt.close()
    
    # 2. Cell type proportions visualization
    print("Creating cell type proportions visualization...")
    
    plt.figure(figsize=(12, 10))
    
    # 2.1 Average cell type proportions (bar plot)
    plt.subplot(2, 1, 1)
    mean_proportions = cell_fractions.mean()
    mean_proportions.plot(kind='bar', color='teal')
    plt.title("Average Cell Type Proportions", fontsize=14)
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha='right')
    
    # 2.2 Cell type proportions per sample (stacked bar)
    plt.subplot(2, 1, 2)
    cell_fractions.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')
    plt.title("Cell Type Proportions by Sample", fontsize=14)
    plt.xlabel("Sample")
    plt.ylabel("Proportion")
    plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/cell_type_proportions.png", dpi=300)
    plt.close()
    
    # 3. Comparison of signature matrix vs. estimated expression
    print("Creating signature vs. estimated expression comparison...")
    
    # Get common genes
    common_genes = list(set(cts_expression.keys()) & set(signature.index))
    
    # For each gene, compare signature values with estimated values
    plt.figure(figsize=(15, 12))
    
    for i, gene in enumerate(common_genes):
        plt.subplot(3, 2, i+1)
        
        # Get data
        prior_values = signature.loc[gene]
        
        # Average estimates across samples
        est_values = cts_expression[gene].mean()
        
        # Create DataFrame for plotting
        compare_df = pd.DataFrame({
            'Signature (Prior)': prior_values,
            'Estimated (Posterior)': est_values
        })
        
        # Plot
        compare_df.plot(kind='bar', ax=plt.gca())
        plt.title(f"Gene: {gene}")
        plt.ylabel("Expression")
        plt.xticks(rotation=45, ha='right')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/signature_vs_estimated.png", dpi=300)
    plt.close()
    
    # 4. Distribution of expression values across cell types
    print("Creating expression distribution visualization...")
    
    # Combine all gene estimates
    all_estimates = pd.DataFrame()
    for gene in cts_expression:
        gene_means = cts_expression[gene].mean()
        all_estimates[gene] = gene_means
    
    # Visualize distributions
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=all_estimates.T)
    plt.title("Distribution of Cell Type-Specific Expression Across Genes", fontsize=14)
    plt.ylabel("Expression Level")
    plt.xlabel("Cell Type")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expression_distribution.png", dpi=300)
    plt.close()
    
    print(f"All visualizations saved to '{output_dir}' directory")



def visualize_mcmc_diagnostics(fit, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    print("Creating MCMC diagnostic plots…")

    def get_chain_array(param):
        """
        Returns a 2D array of shape (n_iter, n_chains) for `param`,
        excluding warmup (inc_warmup=False) and unpermuted.
        """
        arr = fit.extract([param], permuted=False, inc_warmup=False)
        # arr.shape == (n_iter, n_chains, 1); squeeze out last dim
        return arr.squeeze(-1)

    # 1) Per-parameter trace/hist/ACF
    for param in ['a[1,1,1]', 'a[1,1,2]', 'a[1,1,3]']:
        try:
            data = get_chain_array(param)    # shape (n_iter, n_chains)
            n_iter, n_chains = data.shape

            fig = plt.figure(figsize=(12, 8))
            gs  = GridSpec(2, 2, figure=fig)

            # Trace plot
            ax1 = fig.add_subplot(gs[0, :])
            for c in range(n_chains):
                ax1.plot(data[:, c], alpha=0.7, label=f"Chain {c+1}")
            ax1.set(title=f"Trace Plot: {param}", xlabel="Iteration", ylabel="Value")
            ax1.legend()

            # Flatten all draws for hist + ACF
            all_draws = data.flatten()

            # Histogram
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.hist(all_draws, bins=30)
            ax2.set(title=f"Posterior Dist: {param}", xlabel="Value", ylabel="Count")

            # Autocorrelation
            ax3 = fig.add_subplot(gs[1, 1])
            plot_acf(all_draws, ax=ax3, lags=50, alpha=0.05)
            ax3.set(title=f"Autocorrelation: {param}")

            fig.tight_layout()
            fname = f"{output_dir}/mcmc_diagnostics_{param.replace('[','_').replace(']','').replace(',','_')}.png"
            fig.savefig(fname, dpi=300)
            plt.close(fig)

        except Exception as e:
            print(f"Error creating diagnostic plots for {param}: {e}")

    # 2) Joint corner plot
    joint_params = ['a[1,1,1]', 'a[1,1,2]', 'a[1,1,3]', 'a[1,1,4]']
    try:
        chain_data = [get_chain_array(p).flatten() for p in joint_params]
        matrix = np.column_stack(chain_data)
        labels = [p.replace('a[', 'CT Expr: ').replace(']', '') for p in joint_params]

        fig = corner.corner(
            matrix,
            labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12}
        )
        fig.savefig(f"{output_dir}/posterior_joint_distribution.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error creating joint posterior plot: {e}")

    print(f"MCMC diagnostic plots saved to '{output_dir}'")

def main():
    output_dir = "results"
    # Load signature matrix
    signature = pd.read_csv("data/Signature_matrix_Darmanis.csv", index_col=0)
    print(f"Loaded signature matrix: {signature.shape}")
    
    # Use a small subset of genes for testing
    np.random.seed(42)
    n_samples = 5
    n_genes = 5  # Using just 5 genes for quick testing
    
    # Select first few genes
    genes_to_use = signature.index[:n_genes].tolist()
    
    # Create test bulk data
    bulk_data = pd.DataFrame(
        np.random.normal(5, 1, (n_genes, n_samples)),
        index=genes_to_use,
        columns=[f'sample_{i}' for i in range(n_samples)]
    )
    
    # Create test cell fractions
    cell_fractions = pd.DataFrame(
        np.random.dirichlet(np.ones(signature.shape[1]), n_samples),
        index=[f'sample_{i}' for i in range(n_samples)],
        columns=signature.columns
    )
    
    # Create priors from signature
    prior_means = {}
    prior_covs = {}
    for gene in genes_to_use:
        prior_means[gene] = signature.loc[gene].values
        # Simple covariance matrix
        prior_covs[gene] = np.eye(signature.shape[1]) * 0.1
    
    # Run bMIND with sequential processing
    print("Running bMIND with sequential processing...")
    cts_expression, fit = run_bmind_sequential(
        bulk_data=bulk_data,
        cell_fractions=cell_fractions,
        prior_means=prior_means,
        prior_covs=prior_covs,
        iter=300,  # Low for testing
        chains=2   # Just 2 chains
    )
    
    # Print results
    print("\nResults:")
    for gene in cts_expression:
        print(f"\nEstimated cell-type specific expression for {gene}:")
        print(cts_expression[gene])

    # Save results to CSV
    for gene in cts_expression:
        cts_expression[gene].to_csv(f"{output_dir}/cts_expression_{gene}.csv")
    
    print(f"\nSaved cell type-specific expression estimates for {len(cts_expression)} genes")
    
    # Generate visualizations
    visualize_cts_expression(cts_expression, signature, cell_fractions, output_dir)
    
    # Generate MCMC diagnostics
    visualize_mcmc_diagnostics(fit, output_dir)
    
    print("\nAnalysis complete. All results and visualizations saved to the 'results' directory.")

if __name__ == "__main__":
    main()