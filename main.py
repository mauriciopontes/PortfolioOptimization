from tools.assetallocation import AssetAllocation
from tools.dataviz import PortfolioVisualizer

def main(ticlers_list=list):
    # Instanciando o investidor
    investor = AssetAllocation(
        profile='Moderado',
        tolerance_risk=0.08,
        assets=ticlers_list,
        risk_free_rate=0.14,
        data_period="ytd",
        annualization_factor=252
    )

    # Visualização do Asset Allocation
    visualizer = PortfolioVisualizer(
        asset_allocation=investor,
        show_formatted_results=True
    )

    # Gerando os gráficos
    visualizer.plot_efficient_frontier(num_portfolios=10000)
    visualizer.plot_portfolio_composition(portfolio_type='market')
    visualizer.plot_portfolio_composition(portfolio_type='risk_controlled')

if __name__ == "__main__":
    main(ticlers_list=[
        '', '', '', '', '', '', ''
    ])