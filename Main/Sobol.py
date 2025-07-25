"""
This method employs numerical techniques to compute first-order and total-order (TO) 
Sobol sensitivity indices, following the approach outlined by Saltelli (2010). 

The procedure is described in detail in Section 4.6 ("How to compute the sensitivity indices") 
on page 164 of the book:

Reference:
    - Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., 
      Gatelli, D., Saisana, M., & Tarantola, S. (2008). *Global Sensitivity 
      Analysis: The Primer*. John Wiley & Sons.
"""

import numpy as np
from scipy.stats import qmc, rankdata
import time

class SobolAnalyzer:
    def __init__(self, base_samples, n_variables, func, lb=None, ub=None):
        self.base_samples = base_samples
        self.n_variables = n_variables
        self.func = func
        if lb is None:
            self.lb = np.zeros(n_variables)
        else:
            self.lb = lb
        if ub is None:
            self.ub = np.ones(n_variables)
        else:
            self.ub = ub
        self.input_data = None
        self.output_data = None
        self.time_list = None
        self.TO = None
        self.TO_Rank = None
        self.FO = None
        self.FO_Rank = None
        self.Vy = None
        self.MUy = None

    def generate_samples(self):
        D = self.n_variables
        N = self.base_samples
        LB = self.lb
        UB = self.ub
        sampler = qmc.LatinHypercube(d=2 * D)
        if type(N) == list:
            baseSample = sampler.random(n=N[0])
        else:
            baseSample = sampler.random(n=N)
        A = baseSample[:, 0:D]
        B = baseSample[:, D:2 * D]
        A = A * (UB - LB) + LB
        B = B * (UB - LB) + LB
        C = []
        for i in range(D):
            xx = B.copy()
            xx[:, i] = A[:, i]
            C.append(xx)
        return A, B, C

    def evaluate_function(self, A, B, C):
        N, D = A.shape
        yA = np.zeros((N, 1))
        yB = np.zeros((N, 1))
        yC = np.zeros((N, D))
        time_list = np.zeros((N, 1))
        for j in range(N):
            st1 = time.time()
            yA[j, 0] = self.func(A[j, :])
            yB[j, 0] = self.func(B[j, :])
            for i in range(D):
                yC[j, i] = self.func(C[i][j, :])
            ft1 = time.time()
            time_list[j, 0] = ft1 - st1
        return yA, yB, yC, time_list

    def compute_indices(self, yA, yB, yC):
        N, D = yC.shape
        z2 = np.vstack([yA, yB])
        f0 = np.mean(z2)
        self.MUy = f0
        Vy = (np.dot(z2.T, z2)[0][0] / (2 * N)) - f0 ** 2
        self.Vy = Vy
        FO = np.zeros((D, 1))
        TO = np.zeros((D, 1))
        for i in range(D):
            FO[i, 0] = ((np.dot(yA.T, yC[:, i]) / N) - (np.mean(yA) * np.mean(yB))) / Vy
            TO[i, 0] = 1 - ((np.dot(yB.T, yC[:, i]) / N - f0 ** 2) / Vy)
        TO_Rank = (D - rankdata(TO).astype(int)) + 1
        FO_Rank = (D - rankdata(FO).astype(int)) + 1
        self.FO = FO
        self.TO = TO
        self.TO_Rank = TO_Rank
        self.FO_Rank = FO_Rank

    def prepare_data(self, A, B, C, yA, yB, yC):
        N, D = A.shape
        input_data = np.zeros((N * (D + 2), D))
        output_data = np.zeros((N * (D + 2), 1))
        for j in range(D + 2):
            if j == 0:
                input_data[j * N : (j + 1) * N, :] = A
                output_data[j * N : (j + 1) * N, :] = yA
            elif j == 1:
                input_data[j * N : (j + 1) * N, :] = B
                output_data[j * N : (j + 1) * N, :] = yB
            else:
                input_data[j * N : (j + 1) * N, :] = C[j - 2]
                output_data[j * N : (j + 1) * N, 0] = yC[:, j - 2]
        self.input_data = input_data
        self.output_data = output_data

    def perform_analysis(self):
        A, B, C = self.generate_samples()
        yA, yB, yC, time_list = self.evaluate_function(A, B, C)
        self.time_list = time_list
        self.compute_indices(yA, yB, yC)
        self.prepare_data(A, B, C, yA, yB, yC)
        
        