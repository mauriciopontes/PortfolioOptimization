from tools.profile_management import Investor
from dataclasses import dataclass
from typing import Tuple, Dict, Union
import numpy as np
from scipy.optimize import minimize

@dataclass
class AssetAllocation(Investor):
    """Classe para alocação de ativos baseada em otimização de portfolio."""
    INITIAL_WEIGHTS: np.ndarray = None
    CONSTRAINTS: tuple = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    BOUNDS: tuple = None
    
    def __post_init__(self):
        super().__post_init__() 
        if self.risk_free_rate is None:
            raise ValueError("risk_free_rate é obrigatório para AssetAllocation")
        
        n_assets = len(self.assets)
        if self.INITIAL_WEIGHTS is None:
            self.INITIAL_WEIGHTS = np.full(n_assets, 1/n_assets)
        if self.BOUNDS is None:
            self.BOUNDS = tuple((0, 1) for _ in range(n_assets))
        
        self.profile_customer = self.profile
        self.expected_returns = self.returns_annualized
        self.cov_matrix = self.covariance_matrix
        
        self._optimal_weights = self._optimize()
        self._optimal_return, self._optimal_risk = self._portfolio_performance(self._optimal_weights)

    def _portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float]:
        """Calcula o retorno e o risco do portfolio para os pesos fornecidos."""
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        return portfolio_return, portfolio_risk

    def _neg_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calcula o negativo do Sharpe Ratio para uso na otimização."""
        p_return, p_risk = self._portfolio_performance(weights)
        return -(p_return - self.risk_free_rate) / p_risk
    
    def _optimize(self) -> np.ndarray:
        """Otimiza os pesos do portfolio maximizando o Sharpe Ratio."""
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
        """Calcula o Sharpe Ratio do portfolio."""
        return (portfolio_return - self.risk_free_rate) / portfolio_risk
    
    def get_market_portfolio(self, formatted: bool = False) -> Union[Dict[str, float], str]:
        """
        Retorna o portfolio de mercado otimizado com base na maximização do Sharpe Ratio.

        Args:
            formatted (bool): Se True, retorna uma string formatada com os resultados.
                             Se False, retorna um dicionário com os dados brutos. Padrão é False.

        Returns:
            Union[Dict[str, float], str]: Dicionário com os dados do portfolio ou string formatada.
        """
        sharpe_ratio = self._calculate_sharpe(self._optimal_return, self._optimal_risk)
        portfolio_data = {
            'optimal_weights': self._optimal_weights,
            'expected_return': self._optimal_return,
            'risk': self._optimal_risk,
            'sharpe_ratio': sharpe_ratio
        }
        
        if not formatted:
            return portfolio_data
        
        formatted_output = "Portfolio de Mercado:\n"
        formatted_output += "- Pesos Otimizados:\n"
        for asset, weight in zip(self.assets, self._optimal_weights):
            formatted_output += f"  - {asset}: {weight * 100:.2f}%\n"
        formatted_output += f"- Retorno Esperado: {self._optimal_return * 100:.2f}%\n"
        formatted_output += f"- Risco: {self._optimal_risk * 100:.2f}%\n"
        formatted_output += f"- Sharpe Ratio: {sharpe_ratio:.2f}"
        return formatted_output
    
    def get_risk_controlled_portfolio(self, risk_tolerance: float, formatted: bool = False) -> Union[Dict[str, float], str]:
        """
        Retorna o portfolio controlado por risco com base na tolerância ao risco.

        Args:
            risk_tolerance (float): Tolerância ao risco do investidor.
            formatted (bool): Se True, retorna uma string formatada com os resultados.
                             Se False, retorna um dicionário com os dados brutos. Padrão é False.

        Returns:
            Union[Dict[str, float], str]: Dicionário com os dados do portfolio ou string formatada.
        """
        market_proportion = min(1.0, max(0.0, risk_tolerance / self._optimal_risk))
        risk_free_proportion = 1.0 - market_proportion
        
        final_weights = self._optimal_weights * market_proportion
        total_return = (risk_free_proportion * self.risk_free_rate + 
                       market_proportion * self._optimal_return)
        total_risk = market_proportion * self._optimal_risk
        
        sharpe_ratio = self._calculate_sharpe(total_return, total_risk)
        portfolio_data = {
            'optimal_weights': final_weights,
            'expected_return': total_return,
            'risk': total_risk,
            'sharpe_ratio': sharpe_ratio
        }
        
        if not formatted:
            return portfolio_data
        
        formatted_output = "Portfolio Controlado por Risco:\n"
        formatted_output += "- Pesos Otimizados:\n"
        for asset, weight in zip(self.assets, final_weights):
            formatted_output += f"  - {asset}: {weight * 100:.2f}%\n"
        formatted_output += f"- Retorno Esperado: {total_return * 100:.2f}%\n"
        formatted_output += f"- Risco: {total_risk * 100:.2f}%\n"
        formatted_output += f"- Sharpe Ratio: {sharpe_ratio:.2f}"
        return formatted_output