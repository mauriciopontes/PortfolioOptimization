from tools.assetallocation import AssetAllocation
from tools.dataviz import PortfolioVisualizer

def main():
    # Instanciando o investidor
    investor = AssetAllocation(
        profile='Moderado',
        tolerance_risk=0.08,
        assets=['KLBN11.SA', 'SAPR11.SA', 'GGBR4.SA', 'ITSA4.SA', 'TAEE11.SA', 'VBBR3.SA'],
        risk_free_rate=0.02,
        data_period="1y",
        annualization_factor=252
    )

    # Instanciando o visualizador com a opção de exibir resultados formatados
    visualizer = PortfolioVisualizer(
        asset_allocation=investor,
        show_formatted_results=True
    )

    # Gerando os gráficos
    visualizer.plot_efficient_frontier(num_portfolios=10000)
    visualizer.plot_portfolio_composition(portfolio_type='market')
    visualizer.plot_portfolio_composition(portfolio_type='risk_controlled')

if __name__ == "__main__":
    main()