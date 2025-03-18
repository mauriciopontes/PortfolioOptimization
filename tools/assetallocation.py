from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
from scipy.optimize import minimize
from tools.profile_management import Investor

@dataclass
class AssetAllocation(Investor):
    """Classe para alocação de ativos baseada em otimização de portfolio."""
    risk_free_rate: float
    
    # Constantes
    INITIAL_WEIGHTS: np.ndarray = None
    CONSTRAINTS: tuple = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    BOUNDS: tuple = None
    
    def __post_init__(self):
        # Inicialização lazy das constantes baseadas nos assets
        n_assets = len(self.assets)
        if self.INITIAL_WEIGHTS is None:
            self.INITIAL_WEIGHTS = np.full(n_assets, 1/n_assets)
        if self.BOUNDS is None:
            self.BOUNDS = tuple((0, 1) for _ in range(n_assets))
        
        # Cache de atributos usados frequentemente
        self.profile_customer = self.profile
        self.expected_returns = self.returns_anualizeted
        self.cov_matrix = self.matrix_covarience
        
        # Resultados da otimização
        self._optimal_weights = self._optimize()
        self._optimal_return, self._optimal_risk = self._portfolio_performance(self._optimal_weights)

    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calcula retorno e risco do portfolio."""
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_risk

    def _neg_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calcula o negativo do índice Sharpe para otimização."""
        p_return, p_risk = self._portfolio_performance(weights)
        return -(p_return - self.risk_free_rate) / p_risk
    
    def _optimize(self) -> np.ndarray:
        """Realiza a otimização do portfolio usando SLSQP."""
        result = minimize(
            fun=self._neg_sharpe_ratio,
            x0=self.INITIAL_WEIGHTS,
            args=(),
            method='SLSQP',
            bounds=self.BOUNDS,
            constraints=self.CONSTRAINTS,
            options={'disp': False}
        )
        if not result.success:
            raise ValueError(f"Otimização falhou: {result.message}")
        return result.x
    
    def _calculate_sharpe(self, portfolio_return: float, portfolio_risk: float) -> float:
        """Calcula o índice Sharpe."""
        return (portfolio_return - self.risk_free_rate) / portfolio_risk
    
    def get_market_portfolio(self) -> Dict[str, float]:
        """Retorna as métricas do portfolio de mercado otimizado."""
        sharpe_ratio = self._calculate_sharpe(self._optimal_return, self._optimal_risk)
        return {
            'optimal_weights': self._optimal_weights,
            'expected_return': self._optimal_return,
            'risk': self._optimal_risk,
            'sharpe_ratio': sharpe_ratio
        }
    
    def get_risk_controlled_portfolio(self, risk_tolerance: float) -> Dict[str, float]:
        """Retorna portfolio ajustado pela tolerância ao risco."""
        market_proportion = min(1.0, max(0.0, risk_tolerance / self._optimal_risk))
        risk_free_proportion = 1.0 - market_proportion
        
        final_weights = self._optimal_weights * market_proportion
        total_return = (risk_free_proportion * self.risk_free_rate + 
                       market_proportion * self._optimal_return)
        total_risk = market_proportion * self._optimal_risk
        
        sharpe_ratio = self._calculate_sharpe(total_return, total_risk)
        
        return {
            'optimal_weights': final_weights,
            'expected_return': total_return,
            'risk': total_risk,
            'sharpe_ratio': sharpe_ratio
        }