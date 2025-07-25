"""
Example 3: Sensitivity Analysis Comparison Across Multiple Samples and Replications

This module performs a detailed sensitivity analysis on the G-function model, comparing 
RF-PVI and Sobol Total-Order Indices across various sample sizes and replications. 
It computes and visualizes sensitivity indices and input variable rankings using 
the 'ResultsVisualizer' class.

Developed by: Mohammad Khanjani
Date: 2025-07-24  
Version: 1.0.0

Configuration:
    - self.base_sample: List of sample sizes (e.g., [200, 400, 600, 800, 1000]).
    - self.n_variables: Number of input variables (i.e., G-function dimensionality).
    - self.num_replications: Number of replications for each sample size (e.g., 2).
"""

import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy.stats as stats
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

Main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Main'))
sys.path.append(Main_path)

# Custom local imports
from Sobol import SobolAnalyzer
from G_Function import GFunction
from PVIwithOOB import PermutationImportance
from Visual3 import ResultsVisualizer # Import the new visualizer

class SensitivityAnalysisConfig:
    """Configuration class for sensitivity analysis parameters."""
    def __init__(self):
        #============== Inputs:
        self.base_samples = [200, 400, 600, 800, 1000]
        self.n_variables = 8
        self.num_replications = 2 # Number of resampling runs for each base sample
        #==============
        
        
        if self.n_variables == 8 :
            self.coefficients = [0, 0.08, 0.25, 0.55, 1, 3, 15, 90]
        elif self.n_variables == 20:
            self.coefficients = [0.005,    0.02,    0.06,    0.08,    1,
                                2.1,      2.75,    3.1,     3.25,    3.5,
                                8,        10,      12,      14,      16,
                                70,       80,      85,      90,      99]
        else: 
            self.coefficients = eval(input("Give the Coefficients of G-function (e.g., [0, 0.25, 1, 5]): "))
        self.benchmark_type = f'{self.n_variables}d-gfunction'
        self.number_of_runs = [BS * (self.n_variables + 2) for BS in self.base_samples]

class SensitivityAnalyzer:
    """Main class for sensitivity analysis computation with replications."""
    def __init__(self, config=None):
        """Initializes the sensitivity analyzer."""
        self.config = config or SensitivityAnalysisConfig()
        self.variables = [f'P{i+1}' for i in range(self.config.n_variables)]
        
        # Initialize data storage for aggregated results
        n_samples = len(self.config.base_samples)
        n_vars = self.config.n_variables
        
        self.true_indices = None
        self.true_ranks = None
        
        # Storage for index confidence intervals
        self.rf_pvi_mean = np.zeros((n_samples, n_vars))
        self.rf_pvi_ci_lower = np.zeros((n_samples, n_vars))
        self.rf_pvi_ci_upper = np.zeros((n_samples, n_vars))
        
        self.sobol_toi_mean = np.zeros((n_samples, n_vars))
        self.sobol_toi_ci_lower = np.zeros((n_samples, n_vars))
        self.sobol_toi_ci_upper = np.zeros((n_samples, n_vars))
        
        # Storage for rank correctness rates
        self.rank_correctness_rf = np.zeros((n_samples, n_vars))
        self.rank_correctness_sobol = np.zeros((n_samples, n_vars))

    def run_analysis(self):
        """Runs the complete sensitivity analysis for all sample sizes and replications."""
        print(f"Base Sample Sizes: {self.config.base_samples}")
        print(f"Number of Replications per Sample Size: {self.config.num_replications}")
        
        self._compute_analytical_benchmark()
        
        for i, sample_size in enumerate(self.config.base_samples):
            print(f"\n--- Processing Base Sample Size: {sample_size} ---")
            
            # Temporary lists to store results for each replication
            rep_sobol_toi, rep_rf_pvi = [], []
            rep_rank_sobol, rep_rank_rf = [], []
            
            for j in range(self.config.num_replications):
                print(f"  Replication {j+1}/{self.config.num_replications}...")
                input_data, output_data, sobol_toi, sobol_ranks = self._compute_sobol_indices(sample_size)
                rf_pvi, rf_ranks = self._compute_rf_pvi(input_data, output_data)
                
                rep_sobol_toi.append(sobol_toi)
                rep_rf_pvi.append(rf_pvi)
                rep_rank_sobol.append(sobol_ranks)
                rep_rank_rf.append(rf_ranks)
            
            # Aggregate results from replications for the current sample size
            self._aggregate_replication_results(i, rep_rf_pvi, rep_sobol_toi, rep_rank_rf, rep_rank_sobol)
        
        print("\nAnalysis complete. All sensitivity indices and ranks computed and aggregated.")

    def _aggregate_replication_results(self, sample_idx, rf_pvi_reps, sobol_toi_reps, rf_rank_reps, sobol_rank_reps):
        """Calculates mean, CI, and rank correctness from replication data."""
        n_reps = self.config.num_replications
        
        # Convert lists of arrays into 2D numpy arrays (replications x variables)
        rf_pvi_reps = np.array(rf_pvi_reps)
        sobol_toi_reps = np.array(sobol_toi_reps)
        rf_rank_reps = np.array(rf_rank_reps)
        sobol_rank_reps = np.array(sobol_rank_reps)

        # --- Calculate Confidence Intervals for Indices ---
        t_crit = stats.t.ppf(0.975, df=n_reps - 1) # t-critical value for 95% CI

        # RF-PVI
        pvi_mean = np.mean(rf_pvi_reps, axis=0)
        pvi_std_err = np.std(rf_pvi_reps, axis=0, ddof=1) / np.sqrt(n_reps)
        self.rf_pvi_mean[sample_idx, :] = pvi_mean
        self.rf_pvi_ci_lower[sample_idx, :] = pvi_mean - t_crit * pvi_std_err
        self.rf_pvi_ci_upper[sample_idx, :] = pvi_mean + t_crit * pvi_std_err

        # Sobol-TOI
        sobol_mean = np.mean(sobol_toi_reps, axis=0)
        sobol_std_err = np.std(sobol_toi_reps, axis=0, ddof=1) / np.sqrt(n_reps)
        self.sobol_toi_mean[sample_idx, :] = sobol_mean
        self.sobol_toi_ci_lower[sample_idx, :] = sobol_mean - t_crit * sobol_std_err
        self.sobol_toi_ci_upper[sample_idx, :] = sobol_mean + t_crit * sobol_std_err
        
        # --- Calculate Rank Correctness Rate ---
        # Compare ranks from each replication to the true ranks (broadcasts automatically)
        rf_correct_ranks = np.sum(rf_rank_reps == self.true_ranks, axis=0)
        sobol_correct_ranks = np.sum(sobol_rank_reps == self.true_ranks, axis=0)
        
        self.rank_correctness_rf[sample_idx, :] = rf_correct_ranks / n_reps
        self.rank_correctness_sobol[sample_idx, :] = sobol_correct_ranks / n_reps

    def _compute_analytical_benchmark(self):
        """Computes analytical sensitivity indices for benchmarking."""
        print("Loading analytical benchmark for G-function...")
        g_func = GFunction(self.config.coefficients)
        st, true_rank = g_func.analytic_indices()        
        self.true_indices = np.ravel(st.values)
        self.true_ranks = np.array(true_rank, dtype=int)

    def _compute_sobol_indices(self, sample_size):
        """Computes Sobol total-order indices for a given sample size."""
        g_func = GFunction(self.config.coefficients)
        analyzer = SobolAnalyzer(self.config.base_samples, self.config.n_variables, g_func.evaluate)
        analyzer.perform_analysis()
        input_data = analyzer.input_data
        output_data = analyzer.output_data
        sobol_toi = analyzer.TO
        rank_sobol_toi = analyzer.TO_Rank
        return input_data, output_data, np.ravel(sobol_toi), rank_sobol_toi

    def _compute_rf_pvi(self, input_data, output_data):
        """Computes Random Forest Permutation Variable Importance."""
        
        X, y = pd.DataFrame(input_data), pd.DataFrame(output_data)
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X))
                
        rf = RandomForestRegressor()
        rf.fit(X_scaled, y.values.ravel())
        pvi = PermutationImportance.get_feature_importances(rf, X_scaled, y[0], PermutationImportance.compute_oob_mse)        
       
        # pvi = get_permutation_importances(rf, X_scaled, y, calculate_oob_mse)
        rf_pvi = np.ravel((pvi / (2 * y.var()[0])).values)
        
        rf_ranks = (len(rf_pvi) - rankdata(rf_pvi).astype(int)) + 1
        return rf_pvi, rf_ranks

    def save_results(self, filename_prefix="multi_sample_replications"):
        """Saves aggregated analysis results to a CSV file."""
        results_data = []
        for i, sample in enumerate(self.config.base_samples):
            num_run = self.config.number_of_runs[i]
            for j, var in enumerate(self.variables):
                results_data.append({
                    'Base sample': sample,
                    'number_of_runs': num_run,
                    'variable': var,
                    'true_index': self.true_indices[j],
                    'true_rank': self.true_ranks[j],
                    'rf_pvi_mean': self.rf_pvi_mean[i, j],
                    'rf_pvi_ci_lower': self.rf_pvi_ci_lower[i, j],
                    'rf_pvi_ci_upper': self.rf_pvi_ci_upper[i, j],
                    'sobol_toi_mean': self.sobol_toi_mean[i, j],
                    'sobol_toi_ci_lower': self.sobol_toi_ci_lower[i, j],
                    'sobol_toi_ci_upper': self.sobol_toi_ci_upper[i, j],
                    'rf_rank_correctness': self.rank_correctness_rf[i, j],
                    'sobol_rank_correctness': self.rank_correctness_sobol[i, j]
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{filename_prefix}_results.csv", index=False, float_format='%.4f')
        print(f"\nAggregated results saved to {filename_prefix}_results.csv")

def main():
    """Main function to run the sensitivity analysis with replications."""
    print("Starting sensitivity analysis comparison with replications...")
    
    try:
        # 1. Initialize and run the analysis
        config = SensitivityAnalysisConfig()
        analyzer = SensitivityAnalyzer(config)
        analyzer.run_analysis()
        
        # 2. Initialize the visualizer and generate plots
        visualizer = ResultsVisualizer(analyzer)
        print("Generating confidence interval plot for sensitivity indices...")
        visualizer.create_confidence_interval_plot()
        
        print("Generating rank correctness rate plot...")
        visualizer.create_rank_correctness_plot()
        
        # 3. Save aggregated results
        analyzer.save_results()
        
        print("\nProcess finished successfully.")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()