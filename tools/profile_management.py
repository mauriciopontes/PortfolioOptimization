import yfinance as yf
import pandas as pd

class Investor():
    def __init__(self, profile: str, risk: float, assets: list):
        self.profile = profile
        self.risk = risk
        self.assets = assets
        self.datafinance_of_him_assets = self._get_datafinance()
        self.daily_returns = self.datafinance_of_him_assets.pct_change().dropna()
        self._expected_returns_of_him_assets = self.daily_returns.mean() * 252
        self._cov_matrix_of_him_assets = self.daily_returns.cov() * 252

    def _get_datafinance(self, start, end)-> pd.DataFrame:
        return yf.download(self.assets, start=start, end=end)['Close']
    
    @property
    def returns_anualizeted(self):
        return self._expected_returns_of_him_assets
    
    @property
    def matrix_covarience(self):
        return self._cov_matrix_of_him_assets