"""
Example 2: Sensitivity Analysis Comparison Across Multiple Samples

This module extends the G-function sensitivity analysis by evaluating the 
performance of RF-PVI and Sobol Total-Order Indices over a range of sample sizes. 
Sensitivity metrics and rankings are computed and visualized using the 
'ResultsVisualizer' class.

Developed by: Mohammad Khanjani  
Date: 2025-07-24  
Version: 1.0.0

Configuration:
    - self.base_sample: List of different sample sizes to analyze (e.g., [200, 400, 600]).
    - self.n_variables: Define the number of input variables (i.e., the dimensionality 
                        of the G-function).
"""

import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

Main_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Main'))
sys.path.append(Main_path)

# Custom local imports
from Sobol import SobolAnalyzer
from G_Function import GFunction
from PVIwithOOB import PermutationImportance
from Visual2 import ResultsVisualizer # Import the new visualizer


class SensitivityAnalysisConfig:
    """Configuration class for sensitivity analysis parameters."""
    def __init__(self):
        #============== Inputs:
        self.base_samples = [200, 400, 600]
        self.n_variables = 8
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
        self.number_of_runs = [BS*(self.n_variables+2) for BS in self.base_samples]

class SensitivityAnalyzer:
    """Main class for sensitivity analysis computation and comparison."""
    def __init__(self, config=None):
        """Initializes the sensitivity analyzer."""
        self.config = config or SensitivityAnalysisConfig()
        self.variables = [f'P{i+1}' for i in range(self.config.n_variables)]
        
        # Initialize data storage
        self.true_indices = None
        self.true_ranks = None
        self.sobol_toi_list = []
        self.rf_pvi_list = []
        self.rank_sobol_toi_list = []
        self.rank_rf_pvi_list = []

    def run_analysis(self):
        """Runs the complete sensitivity analysis for all sample sizes."""
        print(f"Sample sizes: {self.config.base_samples}")
        
        self._compute_analytical_benchmark()
        
        for sample_size in self.config.base_samples:
            print(f"Computing Sobol-TOI for {sample_size} base samples...")
            input_data, output_data, sobol_toi, sobol_ranks = self._compute_sobol_indices(sample_size)
            rf_pvi, rf_ranks = self._compute_rf_pvi(input_data, output_data)

            # Store results
            self.sobol_toi_list.append(sobol_toi)
            self.rf_pvi_list.append(rf_pvi)
            self.rank_sobol_toi_list.append(sobol_ranks)
            self.rank_rf_pvi_list.append(rf_ranks)
        
        # Convert lists to numpy arrays for easier manipulation
        self.sobol_toi_list = np.array(self.sobol_toi_list)
        self.rf_pvi_list = np.array(self.rf_pvi_list)
        self.rank_sobol_toi_list = np.array(self.rank_sobol_toi_list)
        self.rank_rf_pvi_list = np.array(self.rank_rf_pvi_list)
        
        print("Analysis complete. All sensitivity indices computed.")

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
        print("Collecting input-output data for training ML")
        X, y = pd.DataFrame(input_data), pd.DataFrame(output_data)
        
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X))
        print("Training Random Forest and computing RF-PVI...")        
        rf = RandomForestRegressor()
        rf.fit(X_scaled, y.values.ravel())
        pvi = PermutationImportance.get_feature_importances(rf, X_scaled, y[0], PermutationImportance.compute_oob_mse)        
        # pvi = get_permutation_importances(rf, X_scaled, y, calculate_oob_mse)
        rf_pvi = np.ravel((pvi / (2 * y.var()[0])).values)
        
        rf_ranks = (len(rf_pvi) - rankdata(rf_pvi).astype(int)) + 1
        return rf_pvi, rf_ranks


    def save_results(self, filename_prefix="multi_sample"):
        """Saves analysis results to a CSV file."""
        results_data = []
        for i, sample in enumerate(self.config.base_samples):
            for j, var in enumerate(self.variables):
                results_data.append({
                    'sample_size': sample,
                    'number_of_runs': self.config.number_of_runs, 'variable': var,
                    'true_index': self.true_indices[j], 'true_rank': self.true_ranks[j],
                    'rf_pvi': self.rf_pvi_list[i, j], 'rf_rank': self.rank_rf_pvi_list[i, j],
                    'sobol_toi': self.sobol_toi_list[i, j], 'sobol_rank': self.rank_sobol_toi_list[i, j]
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{filename_prefix}_results.csv", index=False)
        print(f"\nResults saved to {filename_prefix}_results.csv")


def main():
    """Main function to run the sensitivity analysis."""
    print("Starting sensitivity analysis comparison...")

    
    try:
        # 1. Initialize and run the analysis
        config = SensitivityAnalysisConfig()
        analyzer = SensitivityAnalyzer(config)
        analyzer.run_analysis()
        
        # 2. Initialize the visualizer and generate plots
        visualizer = ResultsVisualizer(analyzer)
        print("Generating sensitivity index comparison plot...")
        visualizer.create_sensitivity_plot()
        print("Generating rank comparison plot...")

        visualizer.create_ranking_plot()
        
        # 3. Print summary and save results
        analyzer.save_results()
        
        print("\nProcess finished successfully.")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    main()