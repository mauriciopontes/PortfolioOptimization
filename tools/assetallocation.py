from scipy.optimize import minimize
import numpy as np
from tools.profile_management import Investor

class AssetAllocation(Investor):

    initial_weights = np.array([1/len(Investor.assets)] * len(Investor.assets))
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(len(Investor.assets)))
    
    def __init__(self, risk_free_rate:float):
        self.risk_free_rate = risk_free_rate
        self.profile_costumer = Investor.profile
        self.expected_returns = Investor.returns_anualizeted
        self.cov_matrix = Investor.matrix_covarience
        self._optimal_weights = self._optimizer()
        self._optimize_return, self._optimize_risk = self.portfolio_performance(self._optimal_weights, self.expected_returns, self.cov_matrix)

    def portfolio_performance(weights, expected_returns, cov_matrix):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        return portfolio_return, portfolio_risk

    def _neg_sharpe_ratio(self, weights, expected_returns, cov_matrix):
        p_return, p_risk = self.portfolio_performance(weights, expected_returns, cov_matrix)
        return -(p_return - self.risk_free_rate) / p_risk
    
    def _optimizer(self, initial_weights=initial_weights, bounds=bounds, constraints=constraints):
        optimal = minimize(self._neg_sharpe_ratio, 
                           initial_weights, 
                           args=(self.expected_returns, self.cov_matrix),
                           method='SLSQP', 
                           bounds=bounds, 
                           constraints=constraints)
        return optimal.x
    
    def calculate_sharpe(self, optimize_return, optimize_risk)-> float:
        sharpe_ratio = (optimize_return - self.risk_free_rate) / optimize_risk
        return sharpe_ratio
    
    def show_market_portfolio(self)-> dict:
        mk_port_sharpe_ratio = self.calculate_sharpe(self._optimize_return, self._optimize_risk)
        dict_results = {
            'Optimal_Weights': self._optimal_weights,
            'Expected_Return': self._optimize_return,
            'Risk': self._optimize_risk,
            'Sharpe_ratio': mk_port_sharpe_ratio
            }
        return dict_results
    
    def show_port_riskcontrol(self, tolerance_risk)-> dict:
        proportion_in_riskfree = max(0, min(1, 1-(tolerance_risk/self._optimize_risk)))
        proportion_in_mkport = 1 - proportion_in_riskfree
        final_weights = self._optimal_weights * proportion_in_mkport
        total_return_w_rf = proportion_in_riskfree * self.risk_free_rate + proportion_in_mkport * self._optimize_return
        total_risk_w_rf = proportion_in_mkport * self._optimize_risk
        riskcontrol_sharpe = self.calculate_sharpe(total_return_w_rf, total_risk_w_rf)
        dict_results_riskcontrol = {
            'Optimal_Weights': final_weights,
            'Expected_Return': total_return_w_rf,
            'Risk': total_risk_w_rf,
            'Sharpe_ratio': riskcontrol_sharpe
            }
        return dict_results_riskcontrol