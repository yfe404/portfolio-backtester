# Portfolio Backtester

A Python-based portfolio backtesting tool that analyzes historical performance, calculates key metrics, and generates comprehensive visualizations for investment portfolios.

## Features

- **Historical Performance Analysis**: Backtest your portfolio using real market data
- **Risk Metrics**: Calculate Sharpe ratio, maximum drawdown, volatility, and alpha
- **Benchmark Comparison**: Compare portfolio performance against S&P 500
- **Comprehensive Visualizations**:
  - Portfolio value over time
  - Cumulative returns
  - Drawdown analysis
  - Asset allocation pie chart
  - Monthly returns heatmap
  - Risk-return scatter plot
- **Flexible Configuration**: Easy-to-modify JSON configuration file
- **Detailed Reports**: Year-by-year performance breakdown and individual asset analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yfe404/portfolio-backtester.git
cd portfolio-backtester
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.json` to define your portfolio:

```json
{
  "portfolio": {
    "TICKER": {
      "weight": 10.0,
      "type": "Stock/ETF/Commodity",
      "name": "Asset Name"
    }
  },
  "settings": {
    "start_date": "2020-01-01",
    "end_date": "auto",
    "initial_capital": 100000
  }
}
```

### Configuration Parameters:
- **weight**: Percentage allocation (doesn't need to sum to 100%)
- **type**: Asset classification (Stock, ETF, or Commodity)
- **name**: Display name for the asset
- **start_date**: Backtest start date (YYYY-MM-DD)
- **end_date**: Backtest end date (use "auto" for today)
- **initial_capital**: Starting portfolio value in USD

## Usage

Run the backtester:

```bash
python main.py
```

Or with a custom configuration file:

```bash
python main.py --config my_portfolio.json
```

## Example Portfolio

The default configuration includes a diversified portfolio:
- **Large Cap Tech**: NVDA (11.26%), MSFT (3.99%)
- **Index ETFs**: SPY (21.08%), QQQ (4.44%)
- **International**: MCHI (2.88%), VWO (0.35%)
- **Commodities**: GLD (2.00%), IAU (1.25%)

## Output

The tool generates:

1. **Console Report**: Detailed performance metrics and portfolio composition
2. **Visualizations**: 6-panel chart saved as `portfolio_analysis.png`
3. **Metrics Including**:
   - Total and annualized returns
   - Sharpe ratio
   - Maximum drawdown
   - Volatility
   - Alpha vs S&P 500

## Sample Output

```
============================================================
 PORTFOLIO BACKTEST REPORT
============================================================
Period: 2020-01-01 to 2024-12-28
Initial Capital: $100,000.00
Final Value: $185,234.56

============================================================
 PERFORMANCE METRICS
============================================================
Total Return................. 85.23%
Annualized Return............ 16.72%
Volatility (Annual).......... 18.45%
Sharpe Ratio................ 0.798
Max Drawdown................ -22.34%
S&P 500 Return.............. 65.12%
Alpha vs S&P 500............ 20.11%
```

## Troubleshooting

### Common Issues:

1. **Ticker Not Found**: Some tickers may be delisted or region-specific. Check Yahoo Finance for correct symbols.

2. **No Data Available**: Ensure you have an internet connection and the date range is valid for the ticker.

3. **Installation Issues**: Update pip and try installing packages individually:
   ```bash
   pip install --upgrade pip
   pip install yfinance pandas numpy matplotlib seaborn
   ```

## Advanced Usage

### Custom Date Ranges
Modify `start_date` and `end_date` in `config.json`:
```json
"start_date": "2015-01-01",
"end_date": "2023-12-31"
```

### Adding New Assets
Add tickers to the portfolio section:
```json
"AAPL": {
  "weight": 5.0,
  "type": "Stock",
  "name": "Apple Inc."
}
```

### Rebalancing Analysis
To test different allocations, simply adjust the weights in `config.json` and rerun.

## Data Sources

- **Market Data**: Yahoo Finance via `yfinance`
- **Benchmark**: S&P 500 Index (^GSPC)

## Requirements

- Python 3.7+
- See `requirements.txt` for package dependencies

## Limitations

- Historical data availability varies by ticker
- Does not account for:
  - Trading fees and taxes
  - Dividend reinvestment (unless using adjusted close prices)
  - Rebalancing costs
  - Currency conversion for international assets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Disclaimer

This tool is for educational and research purposes only. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Author

Created by [yfe404](https://github.com/yfe404)

## Acknowledgments

- Data provided by Yahoo Finance
- Built with Python scientific computing stack