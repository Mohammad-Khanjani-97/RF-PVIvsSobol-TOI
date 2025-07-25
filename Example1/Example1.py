"""
Example 1: Sensitivity Analysis Comparison Using a Single Sample

This module performs a sensitivity analysis using the G-function model, comparing 
Random Forest Permutation Variable Importance (RF-PVI) with Sobol Total-Order Indices 
based on a single sample. It computes sensitivity metrics, ranks the input variables, 
and visualizes the results using the 'ResultsVisualizer' class.

Developed by: Mohammad Khanjani
Date: 2025-07-24  
Version: 1.0.0

Configuration:
    - self.base_sample: Specify the base sample to be used.
    - self.n_variables: Define the number of input variables (i.e., the dimensionality 
                        of the G-function).
"""


# === Imports ===
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
from PVIwithOOB import PermutationImportance
from Sobol import SobolAnalyzer
from G_Function import GFunction
from Visual1 import ResultsVisualizer  # Import the visualizer class



# === Configuration ===
class AnalysisConfig:
    """
    Configuration class for the sensitivity analysis computation.
    
    This class centralizes all settings required for the analysis logic.
    Plotting-related settings have been moved to the ResultsVisualizer class.
    """
    def __init__(self):
        #============== Inputs:
        self.base_sample = [2000] # Select a number of sample
        self.n_variables = 8 # Select number of parameters in G-function
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
        self.variables = [f'P{i+1}' for i in range(self.n_variables)]
        self.number_of_runs = self.base_sample[0]*(self.n_variables+2) 

# === Main Analyzer Class ===
class SensitivityComparator:
    """
    Orchestrates the sensitivity analysis computation.

    This class handles data loading, model training, and sensitivity index
    computation, storing the results for visualization.
    """
    def __init__(self, config=None):
        """
        Initializes the SensitivityComparator.

        Args:
            config (AnalysisConfig): A configuration object.
        """
        self.config = config or AnalysisConfig()
        self.results = {}
        # Initialize data storage
        self.true_indices = None
        self.true_ranks = None
        self.sobol_toi_list = []
        self.rf_pvi_list = []
        self.rank_sobol_toi_list = []
        self.rank_rf_pvi_list = []

    def run_analysis(self):
        """
        Executes the full analysis pipeline from data loading to computation.
        """
        print("Starting sensitivity analysis comparison...")
        self._load_analytical_data()
        input_data, output_data = self._load_sobol_results()
        self._compute_rf_pvi(input_data, output_data)
        print("Analysis complete. All sensitivity indices computed.")

    def _load_analytical_data(self):
        """Loads the true analytical sensitivity indices and ranks."""
        print("Loading analytical benchmark for G-function...")
        g_func = GFunction(self.config.coefficients)
        st, true_rank = g_func.analytic_indices()
        self.results['variable'] = self.config.variables
        self.results['true_index'] = np.ravel(st.values)
        self.results['true_rank'] = np.array(true_rank, dtype=int)

    def _load_sobol_results(self):
        """Computes Sobol indices for the specified benchmark."""
        print(f"Computing Sobol-TOI for {self.config.base_sample} base samples...")
        g_func = GFunction(self.config.coefficients)
        analyzer = SobolAnalyzer(self.config.base_sample, self.config.n_variables, g_func.evaluate)
        analyzer.perform_analysis()
        input_data = analyzer.input_data
        output_data = analyzer.output_data
        sobol_toi = analyzer.TO
        rank_sobol_toi = analyzer.TO_Rank
        self.results['sobol_toi'] = np.ravel(sobol_toi)
        self.results['rank_sobol_toi'] = rank_sobol_toi
        return input_data, output_data

    def _compute_rf_pvi(self, input_data, output_data):
        """Trains a Random Forest and computes Permutation Variable Importance."""
        print("Collecting input-output data for training ML")
        X = pd.DataFrame(input_data)
        y = pd.DataFrame(output_data)

        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X))
        print("Training Random Forest and computing RF-PVI...")
        rf = RandomForestRegressor()
        rf.fit(X_scaled, y.values.ravel())
        
        # pvi = get_permutation_importances(rf, X_scaled, pd.Series(y), calculate_oob_mse)
        pvi = PermutationImportance.get_feature_importances(rf, X_scaled, y[0], PermutationImportance.compute_oob_mse)
        rf_pvi = np.ravel((pvi / (2 * y.var()[0])).values)
        self.results['rf_pvi'] = rf_pvi
        self.results['rank_rf_pvi'] = (len(rf_pvi) - rankdata(rf_pvi).astype(int)) + 1

    def save_results(self, filename_prefix="one_sample"):
        results_df = pd.DataFrame(self.results)  #results_data
        results_df.to_csv(f"{filename_prefix}_results.csv", index=False)
        print(f"\nResults saved to {filename_prefix}_results.csv")

def main():
    """
    Main function to initialize, run analysis, and trigger visualization.
    """
    try:
        # 1. Initialize configuration and analyzer
        config = AnalysisConfig()
        analyzer = SensitivityComparator(config)

        # 2. Run the core analysis to compute all values
        analyzer.run_analysis()

        # 3. Initialize the visualizer with the results and config
        visualizer = ResultsVisualizer(analyzer.results, config)
        
        # 4. Generate and display the comparison plots
        visualizer.plot_index_comparison()
        visualizer.plot_rank_comparison()
        
        
        analyzer.save_results()

        print("\nProcess finished successfully.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()