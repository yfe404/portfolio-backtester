#!/bin/bash
# Example usage script for Portfolio Backtester

echo "====================================="
echo "Portfolio Backtester - Usage Examples"
echo "====================================="
echo ""

# Basic usage with default configuration
echo "1. Basic backtest with default configuration:"
echo "   python main.py"
echo ""

# Using custom configuration
echo "2. Using a custom portfolio configuration:"
echo "   python main.py --config examples/conservative_portfolio.json"
echo ""

# Running without plots
echo "3. Running backtest without generating plots:"
echo "   python main.py --no-plot"
echo ""

# Exporting results to CSV
echo "4. Export results to CSV file:"
echo "   python main.py --export results.csv"
echo ""

# Quiet mode
echo "5. Run in quiet mode (minimal output):"
echo "   python main.py --quiet"
echo ""

# Combined options
echo "6. Combined options example:"
echo "   python main.py --config examples/conservative_portfolio.json --export conservative_results.csv --no-plot"
echo ""

# Python script example
echo "7. Using as a Python module:"
cat << 'EOF'

from portfolio_backtest import PortfolioBacktester
from datetime import datetime

# Define your portfolio
portfolio = {
    'AAPL': {'weight': 30.0, 'type': 'Stock', 'name': 'Apple'},
    'GOOGL': {'weight': 25.0, 'type': 'Stock', 'name': 'Google'},
    'AMZN': {'weight': 25.0, 'type': 'Stock', 'name': 'Amazon'},
    'BTC-USD': {'weight': 10.0, 'type': 'Crypto', 'name': 'Bitcoin'},
    'GLD': {'weight': 10.0, 'type': 'Commodity', 'name': 'Gold'}
}

# Create backtester
backtester = PortfolioBacktester(
    portfolio_dict=portfolio,
    start_date='2021-01-01',
    end_date=datetime.now().strftime('%Y-%m-%d'),
    initial_capital=100000,
    verbose=True
)

# Run backtest
backtester.run()

# Generate report
backtester.generate_report()

# Save visualization
backtester.plot_performance(save_path='my_portfolio.png')

# Export to CSV
backtester.export_results('my_results.csv')

# Access metrics directly
metrics = backtester.metrics
print(f"Total Return: {metrics['Total Return']}")
print(f"Sharpe Ratio: {metrics['Sharpe Ratio']}")
EOF

echo ""
echo "For more information, see the README.md file"