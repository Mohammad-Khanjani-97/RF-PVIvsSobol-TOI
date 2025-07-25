"""
Visualization Module for Multi-Sample Sensitivity Analysis

This module provides the 'ResultsVisualizer' class, which is designed to
take the results from the SensitivityAnalyzer and generate professional,
publication-ready plots showing trends across multiple sample sizes.

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import math

class ResultsVisualizer:
    """Handles all visualization for the multi-sample sensitivity analysis."""
    
    def __init__(self, analyzer):
        """
        Initializes the visualizer with the completed analyzer object.

        Args:
            analyzer (SensitivityAnalyzer): The analyzer instance after
                                            run_analysis() has been called.
        """
        self.analyzer = analyzer
        
        # Define plotting styles directly within the visualizer
        self.colors = {
            'rf_pvi': '#3498DB', 'sobol': '#E74C3C', 'true': '#2C3E50',
            'grid': '#BDC3C7', 'spine': '#34495E', 'face': '#FAFAFA'
        }
        self.plot_params = {
            'font.size': 11, 'font.family': 'serif',
            'axes.labelsize': 12, 'axes.titlesize': 14,
            'xtick.labelsize': 10, 'ytick.labelsize': 10,
            'legend.fontsize': 10, 'figure.titlesize': 16,
            'axes.linewidth': 1.2, 'grid.linewidth': 0.8,
            'lines.linewidth': 2.5, 'lines.markersize': 7
        }
        plt.rcParams.update(self.plot_params)



    
    def create_sensitivity_plot(self):
        """Creates the sensitivity indices comparison plot across sample sizes."""
        fig = self._create_figure_grid('Total-Order Sensitivity Comparison Across Number of Runs')
        
        for i in range(self.analyzer.config.n_variables):
            ax = fig.add_subplot(fig.gs[i // fig.cols, i % fig.cols])
            
            ax.plot(self.analyzer.config.number_of_runs, self.analyzer.rf_pvi_list[:, i], label='RF-PVI', color=self.colors['rf_pvi'], marker='o', markerfacecolor='white', markeredgewidth=2)
            ax.plot(self.analyzer.config.number_of_runs, self.analyzer.sobol_toi_list[:, i], label='Sobol-TOI', color=self.colors['sobol'], marker='s', markerfacecolor='white', markeredgewidth=2)
            ax.axhline(y=self.analyzer.true_indices[i], label='True', color=self.colors['true'], linestyle='--')
            
            self._format_sensitivity_axes(ax, i)
        plt.savefig('TO multi sample.jpg',dpi=300,bbox_inches='tight')        
        plt.show()
    
    def create_ranking_plot(self):
        """Creates the parameter ranking comparison plot across sample sizes."""
        fig = self._create_figure_grid('Parameter Importance Rankings Across Number of Runs')
        
        for i in range(self.analyzer.config.n_variables):
            ax = fig.add_subplot(fig.gs[i // fig.cols, i % fig.cols])
            
            ax.plot(self.analyzer.config.number_of_runs, self.analyzer.rank_rf_pvi_list[:, i], label='RF-PVI', color=self.colors['rf_pvi'], marker='o', markerfacecolor='white', markeredgewidth=2)
            ax.plot(self.analyzer.config.number_of_runs, self.analyzer.rank_sobol_toi_list[:, i], label='Sobol-TOI', color=self.colors['sobol'], marker='s', markerfacecolor='white', markeredgewidth=2)
            ax.axhline(y=self.analyzer.true_ranks[i], label='True', color=self.colors['true'], linestyle='--')
            
            self._add_rank_annotations(ax, i)
            self._format_ranking_axes(ax, i)
            
            # Dynamic y-tick labels for ranking plot
            max_rank = self.analyzer.config.n_variables
            if max_rank <= 10:
                ax.set_yticks(np.arange(1, max_rank + 1, 2))  # Step of 2 for small n
            else:
                ax.set_yticks(np.arange(1, max_rank + 1, max(1, math.ceil(max_rank / 5))))  # Sparse ticks for large n
        plt.savefig('Rank multi sample.jpg',dpi=300,bbox_inches='tight')
        plt.show()
    
    def _create_figure_grid(self, title):
        """Helper to create a standard figure and GridSpec layout."""
        n_vars = self.analyzer.config.n_variables
        cols = math.ceil(math.sqrt(n_vars))
        rows = math.ceil(n_vars / cols)
        
        # Adjusted top margin to prevent overlap
        fig = plt.figure(figsize=(cols * 4, rows * 3))
        fig.patch.set_facecolor('white')
        # fig.gs = fig.add_gridspec(rows, cols, hspace=0.65, wspace=0.35, top=0.93, bottom=0.08, left=0.06, right=0.98)
        fig.gs = fig.add_gridspec(rows, cols, hspace=0.65, wspace=0.35)
        fig.suptitle(title, fontsize=18, fontweight='bold')
        fig.cols = cols
        return fig
    


    def _format_sensitivity_axes(self, ax, param_index):
        """Formats axes for the sensitivity index plots."""
        data_min = min(self.analyzer.rf_pvi_list[:, param_index].min(), self.analyzer.sobol_toi_list[:, param_index].min(), self.analyzer.true_indices[param_index])
        data_max = max(self.analyzer.rf_pvi_list[:, param_index].max(), self.analyzer.sobol_toi_list[:, param_index].max(), self.analyzer.true_indices[param_index])
        margin = (data_max - data_min) * 0.15
        ax.set_ylim(data_min - margin, data_max + margin)
        
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
        ax.set_title(f'Parameter {self.analyzer.variables[param_index]}', fontweight='bold', pad=15)
        ax.set_xlabel('Number of Runs')
        ax.set_ylabel('Total-Order (TO)')
        self._apply_common_styling(ax, param_index)
    
    def _format_ranking_axes(self, ax, param_index):
        """Formats axes for the ranking plots."""
        n_vars = self.analyzer.config.n_variables
        ax.set_yticks(range(1, n_vars + 1))
        ax.set_yticklabels([f'{r}' for r in range(1, n_vars + 1)])
        ax.invert_yaxis()
        ax.set_ylim(n_vars + 0.5, 0.5)
        
        ax.set_title(f'Parameter {self.analyzer.variables[param_index]}', fontweight='bold', pad=15)
        ax.set_xlabel('Number of Runs')
        ax.set_ylabel('Sensitivity Rank')
        self._apply_common_styling(ax, param_index, legend_loc='lower right')
    
    def _add_rank_annotations(self, ax, param_index):
        """Adds rank value annotations to the ranking plot."""
        for j, sample in enumerate(self.analyzer.config.number_of_runs):
            rank_rf = self.analyzer.rank_rf_pvi_list[j, param_index]
            rank_sobol = self.analyzer.rank_sobol_toi_list[j, param_index]
            
            ax.annotate(f'{int(rank_rf)}', (sample, rank_rf), xytext=(0, 12), textcoords='offset points', fontsize=9, color=self.colors['rf_pvi'], ha='center', fontweight='bold')
            ax.annotate(f'{int(rank_sobol)}', (sample, rank_sobol), xytext=(0, -18), textcoords='offset points', fontsize=9, color=self.colors['sobol'], ha='center', fontweight='bold')
    
    def _apply_common_styling(self, ax, param_index, legend_loc='upper right'):
        """Applies common styling to subplots."""
        ax.grid(True, linestyle=':', alpha=0.6, color=self.colors['grid'])
        ax.set_axisbelow(True)
        ax.set_facecolor(self.colors['face'])
        
        ax.set_xticks(self.analyzer.config.number_of_runs)
        ax.set_xticklabels([f'{x:,}' for x in self.analyzer.config.number_of_runs], rotation=30)
        
        for spine in ax.spines.values():
            spine.set_color(self.colors['spine'])

        if param_index == 0:
            legend = ax.legend(loc=legend_loc, frameon=True, fancybox=True, shadow=True)
            legend.get_frame().set_alpha(0.95)