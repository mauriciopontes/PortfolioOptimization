from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tools.assetallocation import AssetAllocation

@dataclass
class PortfolioVisualizer:
    """Classe para visualização de portfolios otimizados."""
    asset_allocation: AssetAllocation
    show_formatted_results: bool = False
    
    def __post_init__(self):
        self.market_portfolio = self.asset_allocation.get_market_portfolio(formatted=False)
        self.risk_controlled_portfolio = self.asset_allocation.get_risk_controlled_portfolio(
            self.asset_allocation.tolerance_risk, formatted=False
        )
        self.risk_free_rate = self.asset_allocation.risk_free_rate
        self.assets = self.asset_allocation.assets

        if self.show_formatted_results:
            print(self.asset_allocation.get_market_portfolio(formatted=True))
            print(self.asset_allocation.get_risk_controlled_portfolio(
                self.asset_allocation.tolerance_risk, formatted=True
            ))

    def plot_efficient_frontier(self, num_portfolios: int = 100) -> None:
        """Plota a fronteira eficiente e a linha de alocação de capital com barra de cor do Índice de Sharpe."""
        risks = []
        returns = []
        sharpe_ratios = []
        
        # Calcular risco, retorno e Sharpe Ratio para cada portfolio simulado
        for _ in range(num_portfolios):
            weights = np.random.random(len(self.assets))
            weights /= np.sum(weights)
            ret, risk = self.asset_allocation._portfolio_performance(weights)
            sharpe = (ret - self.risk_free_rate) / risk  # Calcular o Sharpe Ratio
            returns.append(ret * 100)  # Converter para porcentagem
            risks.append(risk * 100)   # Converter para porcentagem
            sharpe_ratios.append(sharpe)

        market_risk = self.market_portfolio['risk'] * 100
        market_return = self.market_portfolio['expected_return'] * 100
        risk_controlled_risk = self.risk_controlled_portfolio['risk'] * 100
        risk_controlled_return = self.risk_controlled_portfolio['expected_return'] * 100
        risk_free_rate = self.risk_free_rate * 100

        # Ajustar o intervalo da CAL para o risco máximo dos portfolios simulados
        max_risk = max(risks) * 1.1  # 10% a mais que o risco máximo
        cal_risks = np.linspace(0, max_risk, 100)
        cal_returns = risk_free_rate + (market_return - risk_free_rate) / market_risk * cal_risks

        plt.figure(figsize=(10, 6))
        # Plotar os portfolios possíveis com cores baseadas no Sharpe Ratio
        scatter = plt.scatter(risks, returns, c=sharpe_ratios, cmap='viridis', alpha=0.4, label='Portfolios Possíveis')
        plt.colorbar(scatter, label='Índice de Sharpe')
        
        plt.plot(cal_risks, cal_returns, 'r--', label='Linha de Alocação de Capital (CAL)')
        plt.scatter(market_risk, market_return, c='green', marker='*', s=200, 
                    label=f'Portfolio de Mercado ({market_return:.2f}%, {market_risk:.2f}%)')
        plt.scatter(risk_controlled_risk, risk_controlled_return, c='orange', marker='o', s=100, 
                    label=f'Portfolio Controlado por Risco ({risk_controlled_return:.2f}%, {risk_controlled_risk:.2f}%)')
        plt.scatter(0, risk_free_rate, c='black', marker='.', 
                    label=f'Ativo Livre de Risco ({risk_free_rate:.2f}%)')

        plt.title('Fronteira Eficiente e Linha de Alocação de Capital')
        plt.xlabel('Risco (Volatilidade) (%)')
        plt.ylabel('Retorno Esperado (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_portfolio_composition(self, portfolio_type: str = 'market') -> None:
        """Plota a composição do portfolio (gráfico de pizza)."""
        if portfolio_type == 'market':
            weights = self.market_portfolio['optimal_weights']
            title = 'Composição do Portfolio de Mercado'
        elif portfolio_type == 'risk_controlled':
            weights = self.risk_controlled_portfolio['optimal_weights']
            title = 'Composição do Portfolio Controlado por Risco'
        else:
            raise ValueError("portfolio_type deve ser 'market' ou 'risk_controlled'")

        labels = [asset for asset, weight in zip(self.assets, weights) if weight > 0.0001]
        sizes = [weight for weight in weights if weight > 0.0001]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))  # Cores diferentes para cada ativo

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3))
        ax.axis('equal')
        plt.title(title)
        plt.show()