from scipy.optimize import minimize
import numpy as np
from tools.profile_management import Investor
from tools.bracenselic import Selic

class AssetAllocation(Investor, Selic):

    risk_free_rate = Selic.rate
    
    def __init__(self, returns):
        self.profile_costumer = Investor.profile
        self.expected_returns = Investor.returns_anualizeted
        self.cov_matrix = Investor.matrix_covarience
        self._initial_weights = np.array([1/len(Investor.assets)] * len(Investor.assets))
        self._optimize_return
        self._optimize_risk