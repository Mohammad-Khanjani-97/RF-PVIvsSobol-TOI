"""
This module computes the output of the G-function and its corresponding analytical 
sensitivity indices. The G-function was originally introduced by Saltelli (1995) 
as a benchmark for global sensitivity analysis.

Reference:
    - Saltelli, A., & Sobol, I. M. (1995). About the use of rank transformation in 
      sensitivity analysis of model output. *Reliability Engineering & System Safety*, 
      50(3), 225–239.
"""


import pandas as pd

class GFunction:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def evaluate(self, matrix):
        prod = 1
        for i in range(len(matrix)):
            xi = matrix[i]
            ai = self.coefficients[i]
            new1 = abs(4 * float(xi) - 2) + float(ai)
            new2 = 1 + float(ai)
            prod *= new1 / new2
        return prod

    def analytic_indices(self):
        coefficients = self.coefficients
        True_Rank = list(range(1, len(coefficients) + 1))
        ST = {}
        var_num = [f'Parameter {i}' for i in range(1, len(coefficients) + 1)]
        for i in range(len(coefficients)):
            Numerator = 1
            Denominator = 1
            for j in range(len(coefficients)):
                if j != i:
                    Numerator *= (1 + (1 / (3 * (1 + coefficients[j]) ** 2)))
                Denominator *= (1 + (1 / (3 * (1 + coefficients[j]) ** 2)))
            result = ((1 / (3 * (1 + coefficients[i]) ** 2)) * Numerator) / (Denominator - 1)
            ST[var_num[i]] = [result]
        st_df = pd.DataFrame(ST).T
        return st_df, True_Rank
    