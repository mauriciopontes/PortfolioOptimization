from typing import List
import yfinance as yf
import pandas as pd
from dataclasses import dataclass

@dataclass
class Investor:
    """Classe que representa um investidor e seus dados financeiros."""
    profile: str
    tolerance_risk: float
    assets: List[str]
    
    # Período padrão de dados (pode ser sobrescrito)
    DATA_PERIOD: str = "2y"
    ANNUALIZATION_FACTOR: int = 252
    
    def __post_init__(self):
        """Inicialização pós-dataclass com validações e cálculos."""
        self._validate_inputs()
        self._data = self._fetch_financial_data()
        self._daily_returns = self._calculate_daily_returns()
        self._annual_returns = self._calculate_annual_returns()
        self._cov_matrix = self._calculate_cov_matrix()

    def _validate_inputs(self) -> None:
        """Valida os parâmetros de entrada."""
        if not isinstance(self.profile, str) or not self.profile.strip():
            raise ValueError("Profile deve ser uma string não vazia")
        if not isinstance(self.tolerance_risk, (int, float)) or self.tolerance_risk < 0:
            raise ValueError("Tolerance_risk deve ser um número não negativo")
        if not isinstance(self.assets, list) or not self.assets:
            raise ValueError("Assets deve ser uma lista não vazia")
        if not all(isinstance(asset, str) for asset in self.assets):
            raise ValueError("Todos os assets devem ser strings")

    def _fetch_financial_data(self) -> pd.DataFrame:
        """Obtém dados financeiros dos ativos usando yfinance."""
        try:
            data = yf.download(self.assets, period=self.DATA_PERIOD, progress=False)['Close']
            if data.empty:
                raise ValueError("Nenhum dado retornado para os ativos especificados")
            return data
        except Exception as e:
            raise RuntimeError(f"Falha ao obter dados financeiros: {str(e)}")

    def _calculate_daily_returns(self) -> pd.DataFrame:
        """Calcula os retornos diários dos ativos."""
        return self._data.pct_change().dropna()

    def _calculate_annual_returns(self) -> pd.Series:
        """Calcula os retornos anualizados esperados."""
        return self._daily_returns.mean() * self.ANNUALIZATION_FACTOR

    def _calculate_cov_matrix(self) -> pd.DataFrame:
        """Calcula a matriz de covariância anualizada."""
        return self._daily_returns.cov() * self.ANNUALIZATION_FACTOR

    @property
    def returns_annualized(self) -> pd.Series:
        """Retorna os retornos anualizados dos ativos."""
        return self._annual_returns

    @property
    def covariance_matrix(self) -> pd.DataFrame:
        """Retorna a matriz de covariância dos ativos."""
        return self._cov_matrix

    @property
    def daily_returns(self) -> pd.DataFrame:
        """Retorna os retornos diários dos ativos."""
        return self._daily_returns

    @property
    def financial_data(self) -> pd.DataFrame:
        """Retorna os dados financeiros brutos dos ativos."""
        return self._data