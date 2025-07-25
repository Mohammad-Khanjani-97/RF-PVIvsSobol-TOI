# RF-PVI vs Sobol-TOI: Sensitivity Analysis Comparison

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)


A comprehensive Python package for comparing **Random Forest Permutation Variable Importance (RF-PVI)** with **Sobol Total-Order Indices (Sobol-TOI)** in global sensitivity analysis. This implementation explores the theoretical connection between machine learning-based importance measures and variance-based sensitivity indices using the G-function benchmark.

## Overview

This package implements and compares two prominent sensitivity analysis methods:

- **RF-PVI**: Machine learning-based importance using Random Forests with Out-of-Bag (OOB) samples
- **Sobol-TOI**: Variance-based global sensitivity analysis using Sobol indices

The comparison is performed using the **G-function**, a well-established benchmark in sensitivity analysis literature, providing both analytical and numerical validation of results.

## Key Features

- **Three Progressive Examples**: From single-sample analysis to multi-sample with replications
- **Analytical Benchmarking**: Compare against known analytical G-function sensitivity indices
- **Professional Visualizations**: Publication-ready plots with confidence intervals
- **Robust Implementation**: Handles various sample sizes and dimensionalities
- **Statistical Analysis**: Includes rank correctness rates and confidence intervals

## Repository Structure

```
RF-PVIvsSobol-TOI/
├── Main/
│   ├── G_Function.py          # G-function implementation with analytical indices
│   ├── PVIwithOOB.py          # Random Forest PVI with OOB estimation
│   └── Sobol.py               # Sobol indices computation
├── Example1/
│   ├── Example1.py            # Single sample analysis
│   └── Visual1.py             # Visualization for single sample
├── Example2/
│   ├── Example2.py            # Multi-sample analysis
│   └── Visual2.py             # Visualization for multi-sample
├── Example3/
│   ├── Example3.py            # Multi-sample with replications
│   └── Visual3.py             # Advanced visualization with CI
└── README.md
```

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Mohammad-Khanjani-97/RF-PVIvsSobol-TOI.git
cd RF-PVIvsSobol-TOI
```

2. Run your first analysis:
```bash
cd Example1
python Example1.py
```

## Examples

### Example 1: Single Sample Analysis
Compares RF-PVI and Sobol-TOI using a single sample size, providing basic sensitivity index and ranking comparisons.

**Configuration:**
```python
self.base_sample = [2000]    # Sample size
self.n_variables = 8         # G-function dimensionality
```

**Output:**
- Sensitivity index comparison plot
- Parameter ranking comparison plot
- Numerical results in CSV format

### Example 2: Multi-Sample Analysis
Evaluates method performance across multiple sample sizes to study convergence behavior.

**Configuration:**
```python
self.base_samples = [200, 400, 600]  # Multiple sample sizes
self.n_variables = 8                 # G-function dimensionality
```

**Output:**
- Convergence plots across sample sizes
- Individual parameter sensitivity trends
- Comprehensive multi-sample results

### Example 3: Multi-Sample with Replications
Advanced analysis including statistical replications for confidence interval estimation and rank correctness assessment.

**Configuration:**
```python
self.base_samples = [200, 400, 600, 800, 1000]  # Sample sizes
self.n_variables = 8                             # Dimensionality
self.num_replications = 2                        # Statistical replications
```

**Output:**
- 95% confidence intervals for sensitivity indices
- Rank correctness rates by sensitivity groups
- Statistical significance assessment

## 🔧 Core Components

### G-Function (`Main/G_Function.py`)
Implementation of the Saltelli (1995) G-function benchmark with analytical sensitivity indices:

```python
from G_Function import GFunction

# Initialize with coefficients
g_func = GFunction([0, 0.08, 0.25, 0.55, 1, 3, 15, 90])

# Evaluate function
result = g_func.evaluate(input_vector)

# Get analytical indices
indices, ranks = g_func.analytic_indices()
```

### RF-PVI with OOB (`Main/PVIwithOOB.py`)
Random Forest Permutation Variable Importance using Out-of-Bag samples:

```python
from PVIwithOOB import PermutationImportance

# Compute feature importances
importance_df = PermutationImportance.get_feature_importances(
    model, X, y, PermutationImportance.compute_oob_mse
)
```

### Sobol Analysis (`Main/Sobol.py`)
Numerical computation of Sobol sensitivity indices following Saltelli (2008):

```python
from Sobol import SobolAnalyzer

# Initialize analyzer
analyzer = SobolAnalyzer(sample_size, n_variables, function)

# Perform analysis
analyzer.perform_analysis()
total_order_indices = analyzer.TO
```

## Theoretical Background

This package explores the theoretical connection between RF-PVI and Sobol-TOI as established by:

- **Gregorutti et al. (2017)**: Correlation and variable importance in random forests
- **Wei et al. (2015)**: Comprehensive comparison of variable importance techniques
- **Saltelli (2008)**: Global Sensitivity Analysis: The Primer

The G-function provides an ideal benchmark because:
- **Analytical solution available** for validation
- **Controllable parameter importance** through coefficients
- **Well-established** in sensitivity analysis literature

## Visualization Features

All examples generate publication-ready plots with:
- **Professional styling** with seaborn aesthetics
- **Confidence intervals** (Example 3)
- **Rank correctness rates** by sensitivity groups
- **High-resolution output** (300 DPI) for publications
- **Customizable color schemes** and layouts

## Configuration Options

### G-Function Dimensionality
- **8D**: `coefficients = [0, 0.08, 0.25, 0.55, 1, 3, 15, 90]`
- **20D**: Pre-defined 20-dimensional coefficients
- **Custom**: User-defined coefficient arrays

### Sample Size Selection
Configure base sample sizes according to your computational budget:
```python
# Light analysis
self.base_samples = [200, 400, 600]

# Comprehensive analysis  
self.base_samples = [200, 400, 600, 800, 1000, 1500, 2000]
```

## Research Applications

This package is suitable for:
- **Method comparison studies** in sensitivity analysis
- **Machine learning interpretability** research
- **Uncertainty quantification** applications
- **Educational purposes** in global sensitivity analysis

## References

1. **Saltelli, A., & Sobol, I. M. (1995)**. About the use of rank transformation in sensitivity analysis of model output. *Reliability Engineering & System Safety*, 50(3), 225–239.

2. **Gregorutti, B., Michel, B., & Saint-Pierre, P. (2017)**. Correlation and variable importance in random forests. *Statistics and Computing*, 27, 659–678.

3. **Wei, P., Lu, Z., & Song, J. (2015)**. A comprehensive comparison of two variable importance analysis techniques in high dimensions. *Environmental Modelling & Software*, 70, 178–190.

4. **Saltelli, A., et al. (2008)**. *Global Sensitivity Analysis: The Primer*. John Wiley & Sons.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by the [`rfpimp` ](https://github.com/parrt/random-forest-importances) library by Terence Parr  
- Built upon established sensitivity analysis frameworks  
- Thanks to the global sensitivity analysis research community




