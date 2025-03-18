from tools.assetallocation import AssetAllocation

def main():
    # Criando um investidor
    investor = AssetAllocation(
        profile="Moderado",
        tolerance_risk=0.05,  # Tolerância ao risco de 5%
        assets=["AAPL", "MSFT", "GOOGL"],  # Ativos do portfolio
        risk_free_rate=0.02  # Taxa livre de risco de 2%
    )

    # Obtendo o portfolio de mercado otimizado
    market_portfolio = investor.get_market_portfolio()
    print("Portfolio de Mercado:")
    print(f"Pesos Otimizados: {market_portfolio['optimal_weights']}")
    print(f"Retorno Esperado: {market_portfolio['expected_return']:.4f}")
    print(f"Risco: {market_portfolio['risk']:.4f}")
    print(f"Sharpe Ratio: {market_portfolio['sharpe_ratio']:.4f}\n")

    # Obtendo o portfolio ajustado pela tolerância ao risco
    risk_controlled_portfolio = investor.get_risk_controlled_portfolio(investor.tolerance_risk)
    print("Portfolio Controlado por Risco:")
    print(f"Pesos Otimizados: {risk_controlled_portfolio['optimal_weights']}")
    print(f"Retorno Esperado: {risk_controlled_portfolio['expected_return']:.4f}")
    print(f"Risco: {risk_controlled_portfolio['risk']:.4f}")
    print(f"Sharpe Ratio: {risk_controlled_portfolio['sharpe_ratio']:.4f}")

if __name__ == "__main__":
    main()