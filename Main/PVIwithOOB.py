"""
This module implements Permutation Variable Importance (PVI) using Random Forests (RF),
where the importance of each variable is assessed by measuring the  
Mean Squared Error (MSE) across trees using Out-of-Bag (OOB) samples.

The main objective is to explore the theoretical connection between RF-based PVI 
and Sobol Total-Order Indices (Sobol-TOI). This relationship was first introduced 
by Gregorutti et al. (2017) and Wei et al. (2015), who identified a link between 
these two sensitivity analysis approaches.

References:
    - Gregorutti, B., Michel, B., & Saint-Pierre, P. (2017). Correlation and variable 
      importance in random forests. *Statistics and Computing*, 27, 659–678.
    - Wei, P., Lu, Z., & Song, J. (2015). A comprehensive comparison of two variable 
      importance analysis techniques in high dimensions: Application to an 
      environmental multi-indicators system. *Environmental Modelling & Software*, 70, 178–190.

The implementation was inspired in part by the 'rfpimp' library developed by Terence Parr:
    - https://github.com/parrt/random-forest-importances
"""


import pandas as pd
import numpy as np
from typing import Optional, Union, Callable, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from packaging import version
import sklearn

class PermutationImportance:
    """Class to compute permutation feature importance and OOB scores for RandomForest models."""

    @staticmethod
    def _subsample_data(
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int,
        weights: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Subsample the dataset to the specified number of samples."""
        if X.empty or y.empty:
            raise ValueError("Input DataFrame X or Series y is empty.")
        
        sample_size = min(n_samples, len(X)) if n_samples > 0 else len(X)
        
        if sample_size < len(X):
            indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X.iloc[indices].copy(deep=False)
            y_sample = y.iloc[indices].copy(deep=False)
            weights_sample = weights.iloc[indices].copy(deep=False) if weights is not None else None
        else:
            X_sample = X.copy(deep=False)
            y_sample = y.copy(deep=False)
            weights_sample = weights.copy(deep=False) if weights is not None else None
            
        return X_sample, y_sample, weights_sample

    @staticmethod
    def _compute_permutation_importance(
        model: RandomForestRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Callable,
        n_samples: int = 5000
    ) -> np.ndarray:
        """Compute raw permutation importance scores for each feature."""
        if X.empty:
            raise ValueError("Input DataFrame X is empty.")
            
        X_sample, y_sample, _ = PermutationImportance._subsample_data(X, y, n_samples)
        
        if not hasattr(model, 'estimators_'):
            model.fit(X_sample, y_sample)
            
        baseline_score = metric(model, X_sample, y_sample)
        importances = []
        
        for feature in X_sample.columns:
            original_column = X_sample[feature].copy()
            X_sample[feature] = np.random.permutation(X_sample[feature].values)
            score = metric(model, X_sample, y_sample)
            X_sample[feature] = original_column
            importances.append(score - baseline_score)
            
        return np.array(importances)

    @staticmethod
    def get_feature_importances(
        model: RandomForestRegressor,
        X: pd.DataFrame,
        y: pd.Series,
        metric: Callable,
        n_samples: int = 5000
    ) -> pd.DataFrame:
        """Return a DataFrame with feature importance scores."""
        importances = PermutationImportance._compute_permutation_importance(model, X, y, metric, n_samples)
        return pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        }).set_index('Feature')

    @staticmethod
    def _get_oob_indices(tree: object, n_samples: int) -> np.ndarray:
        """Retrieve out-of-bag sample indices for a single tree."""
        from sklearn.ensemble._forest import _generate_unsampled_indices
        
        if version.parse(sklearn.__version__) >= version.parse("0.24"):
            from sklearn.ensemble._forest import _get_n_samples_bootstrap
        elif version.parse(sklearn.__version__) >= version.parse("0.22"):
            from sklearn.ensemble.forest import _get_n_samples_bootstrap
        else:
            return _generate_unsampled_indices(tree.random_state, n_samples)
            
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, n_samples)
        return _generate_unsampled_indices(tree.random_state, n_samples, n_samples_bootstrap)

    @staticmethod
    def compute_oob_mse(
        model: RandomForestRegressor,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> float:
        """Calculate the out-of-bag mean squared error for a RandomForest regressor."""
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        if len(X_array) == 0:
            raise ValueError("Input X is empty.")
            
        mse_scores: List[float] = []
        for tree in model.estimators_:
            oob_indices = PermutationImportance._get_oob_indices(tree, len(X_array))
            if len(oob_indices) == 0:
                continue  # Skip trees with no OOB samples
            predictions = tree.predict(X_array[oob_indices])
            mse = mean_squared_error(y_array[oob_indices], predictions)
            mse_scores.append(mse)
            
        if not mse_scores:
            raise ValueError("No out-of-bag samples available for MSE calculation.")
            
        return float(np.mean(mse_scores))


