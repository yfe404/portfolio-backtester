"""
Portfolio Backtester - Core Module
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from typing import Dict, Optional, Tuple
warnings.filterwarnings('ignore')

class PortfolioBacktester:
    """A comprehensive portfolio backtesting class"""
    
    def __init__(self, portfolio_dict: Dict, start_date: str, end_date: str, 
                 initial_capital: float = 100000, verbose: bool = True):
        """
        Initialize the backtester
        
        Args:
            portfolio_dict: Dictionary of tickers with weight, type, and name
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial investment amount
            verbose: Whether to print detailed progress
        """
        self.portfolio = portfolio_dict
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.verbose = verbose
        self.price_data = {}
        self.returns_data = {}
        self.portfolio_value = None
        self.portfolio_returns = None
        self.benchmark_data = None
        self.metrics = {}
        
    def _print(self, message: str, end='\n'):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message, end=end)
    
    def fetch_data(self) -> None:
        """Fetch historical price data for all tickers"""
        self._print("Fetching historical data...")
        self._print("-" * 50)
        
        successful_tickers = {}
        failed_tickers = []
        
        for ticker, info in self.portfolio.items():
            try:
                self._print(f"Downloading {ticker} ({info['name']})...", end=' ')
                
                # Download data with proper error handling
                stock = yf.Ticker(ticker)
                data = stock.history(start=self.start_date, end=self.end_date, auto_adjust=True)
                
                if not data.empty and 'Close' in data.columns:
                    self.price_data[ticker] = data['Close']
                    successful_tickers[ticker] = info
                    self._print("✓")
                else:
                    self._print("⚠️ No data available")
                    failed_tickers.append(ticker)
                    
            except Exception as e:
                self._print(f"❌ Error: {str(e)[:50]}")
                failed_tickers.append(ticker)
        
        # Update portfolio with successful tickers only
        self.portfolio = successful_tickers
        
        if not self.portfolio:
            raise ValueError("No data could be fetched for any ticker.")
        
        # Normalize weights
        total_weight = sum(info['weight'] for info in self.portfolio.values())
        for ticker in self.portfolio:
            self.portfolio[ticker]['normalized_weight'] = self.portfolio[ticker]['weight'] / total_weight
        
        self._print("-" * 50)
        self._print(f"✓ Successfully loaded {len(self.portfolio)} out of {len(portfolio_dict)} tickers")
        self._print(f"Total weight covered: {total_weight:.2f}%")
        
        if failed_tickers:
            self._print(f"Failed tickers: {', '.join(failed_tickers)}")
        
        # Download benchmark
        self._fetch_benchmark()
    
    def _fetch_benchmark(self) -> None:
        """Fetch S&P 500 benchmark data"""
        try:
            self._print("\nDownloading benchmark (S&P 500)...", end=' ')
            benchmark = yf.Ticker('^GSPC')
            benchmark_data = benchmark.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            
            if not benchmark_data.empty and 'Close' in benchmark_data.columns:
                self.benchmark_data = benchmark_data['Close']
                self._print("✓")
            else:
                self._print("⚠️ Using SPY as benchmark")
                if 'SPY' in self.price_data:
                    self.benchmark_data = self.price_data['SPY']
        except:
            self._print("⚠️ Benchmark failed, using SPY if available")
            if 'SPY' in self.price_data:
                self.benchmark_data = self.price_data['SPY']
    
    def calculate_returns(self) -> None:
        """Calculate daily returns for each asset"""
        self._print("\nCalculating returns...")
        for ticker in self.price_data:
            self.returns_data[ticker] = self.price_data[ticker].pct_change().fillna(0)
    
    def calculate_portfolio_performance(self) -> None:
        """Calculate portfolio value over time"""
        self._print("Calculating portfolio performance...")
        
        # Create aligned dataframe
        prices_df = pd.DataFrame(self.price_data)
        prices_df = prices_df.fillna(method='ffill').fillna(method='bfill')
        prices_df = prices_df.dropna()
        
        if prices_df.empty:
            raise ValueError("No overlapping data periods for portfolio assets")
        
        # Calculate initial shares
        initial_prices = prices_df.iloc[0]
        shares = {}
        
        for ticker in self.portfolio:
            if ticker in initial_prices.index:
                allocation = self.initial_capital * self.portfolio[ticker]['normalized_weight']
                shares[ticker] = allocation / initial_prices[ticker]
        
        # Calculate portfolio value over time
        portfolio_values = []
        for idx, row in prices_df.iterrows():
            daily_value = sum(
                shares.get(ticker, 0) * row.get(ticker, 0) 
                for ticker in self.portfolio if ticker in row.index
            )
            portfolio_values.append(daily_value)
        
        self.portfolio_value = pd.Series(portfolio_values, index=prices_df.index)
        self.portfolio_returns = self.portfolio_value.pct_change().fillna(0)
    
    def calculate_metrics(self) -> Dict:
        """Calculate key performance metrics"""
        self._print("Calculating performance metrics...")
        
        trading_days = 252
        
        # Basic metrics
        total_return = (self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0] - 1) * 100
        years = len(self.portfolio_returns) / trading_days
        
        if years > 0:
            annual_return = (((self.portfolio_value.iloc[-1] / self.portfolio_value.iloc[0]) ** (1/years)) - 1) * 100
        else:
            annual_return = 0
        
        # Risk metrics
        volatility = self.portfolio_returns.std() * np.sqrt(trading_days) * 100
        downside_returns = self.portfolio_returns[self.portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(trading_days) * 100
        
        # Sharpe and Sortino ratios
        risk_free_rate = 2  # 2% annual
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown metrics
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        positive_days = (self.portfolio_returns > 0).sum()
        total_days = len(self.portfolio_returns)
        win_rate = (positive_days / total_days * 100) if total_days > 0 else 0
        
        # Benchmark comparison
        benchmark_return = 0
        alpha = 0
        beta = 0
        
        if self.benchmark_data is not None and len(self.benchmark_data) > 0:
            benchmark_return = (self.benchmark_data.iloc[-1] / self.benchmark_data.iloc[0] - 1) * 100
            alpha = total_return - benchmark_return
            
            # Calculate beta
            benchmark_returns = self.benchmark_data.pct_change().fillna(0)
            benchmark_returns = benchmark_returns.reindex(self.portfolio_returns.index, fill_value=0)
            
            covariance = self.portfolio_returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        self.metrics = {
            'Total Return': f'{total_return:.2f}%',
            'Annualized Return': f'{annual_return:.2f}%',
            'Volatility (Annual)': f'{volatility:.2f}%',
            'Sharpe Ratio': f'{sharpe_ratio:.3f}',
            'Sortino Ratio': f'{sortino_ratio:.3f}',
            'Max Drawdown': f'{max_drawdown:.2f}%',
            'Calmar Ratio': f'{calmar_ratio:.3f}',
            'Win Rate': f'{win_rate:.1f}%',
            'Beta': f'{beta:.3f}',
            'S&P 500 Return': f'{benchmark_return:.2f}%',
            'Alpha vs S&P 500': f'{alpha:.2f}%'
        }
        
        return self.metrics
    
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of portfolio performance"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(self.portfolio_value.index, self.portfolio_value, 
                label='Portfolio', linewidth=2, color='#2E86AB')
        
        if self.benchmark_data is not None and len(self.benchmark_data) > 0:
            benchmark_aligned = self.benchmark_data.reindex(self.portfolio_value.index, method='ffill')
            benchmark_scaled = benchmark_aligned / benchmark_aligned.iloc[0] * self.initial_capital
            ax1.plot(benchmark_scaled.index, benchmark_scaled, 
                    label='S&P 500', linewidth=2, alpha=0.7, color='#A23B72')
        
        ax1.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value ($)')
        ax1.set_xlabel('Date')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Cumulative returns
        ax2 = axes[0, 1]
        cum_returns = (1 + self.portfolio_returns).cumprod() - 1
        ax2.plot(cum_returns.index, cum_returns * 100, 
                label='Portfolio', linewidth=2, color='#2E86AB')
        
        if self.benchmark_data is not None:
            benchmark_returns = self.benchmark_data.pct_change().fillna(0)
            benchmark_returns_aligned = benchmark_returns.reindex(self.portfolio_returns.index, fill_value=0)
            cum_benchmark = (1 + benchmark_returns_aligned).cumprod() - 1
            ax2.plot(cum_benchmark.index, cum_benchmark * 100, 
                    label='S&P 500', linewidth=2, alpha=0.7, color='#A23B72')
        
        ax2.set_title('Cumulative Returns', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.set_xlabel('Date')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # 3. Drawdown
        ax3 = axes[1, 0]
        cumulative = (1 + self.portfolio_returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = ((cumulative - running_max) / running_max) * 100
        ax3.fill_between(drawdown.index, drawdown, 0, 
                         alpha=0.3, color='#E63946', label='Drawdown')
        ax3.plot(drawdown.index, drawdown, linewidth=1, color='#E63946', alpha=0.8)
        ax3.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}%'))
        
        # 4. Asset allocation
        ax4 = axes[1, 1]
        weights = [info['normalized_weight'] for info in self.portfolio.values()]
        labels = [f"{info['name']}\n{info['normalized_weight']*100:.1f}%" 
                 for info in self.portfolio.values()]
        colors = plt.cm.Set3(range(len(weights)))
        wedges, texts, autotexts = ax4.pie(weights, labels=labels, colors=colors, 
                                            autopct='', startangle=90)
        ax4.set_title('Portfolio Allocation', fontsize=12, fontweight='bold')
        
        # 5. Monthly returns heatmap
        ax5 = axes[2, 0]
        try:
            monthly_returns = self.portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
            if len(monthly_returns) > 0:
                monthly_pivot = pd.pivot_table(
                    pd.DataFrame({
                        'Year': monthly_returns.index.year,
                        'Month': monthly_returns.index.month,
                        'Return': monthly_returns.values
                    }),
                    values='Return', index='Month', columns='Year'
                )
                if not monthly_pivot.empty:
                    sns.heatmap(monthly_pivot, annot=True, fmt='.1f', 
                               cmap='RdYlGn', center=0, ax=ax5, 
                               cbar_kws={'label': 'Return (%)'})
                    ax5.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
                    ax5.set_ylabel('Month')
        except:
            ax5.text(0.5, 0.5, 'Insufficient data for monthly heatmap', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Monthly Returns Heatmap', fontsize=12, fontweight='bold')
        
        # 6. Risk-Return scatter
        ax6 = axes[2, 1]
        for ticker in self.portfolio:
            if ticker in self.returns_data and len(self.returns_data[ticker]) > 0:
                annual_ret = self.returns_data[ticker].mean() * 252 * 100
                annual_vol = self.returns_data[ticker].std() * np.sqrt(252) * 100
                ax6.scatter(annual_vol, annual_ret, s=100, alpha=0.6)
                ax6.annotate(self.portfolio[ticker]['name'][:10], 
                           (annual_vol, annual_ret), fontsize=8, alpha=0.8)
        
        # Add portfolio point
        portfolio_annual_ret = self.portfolio_returns.mean() * 252 * 100
        portfolio_annual_vol = self.portfolio_returns.std() * np.sqrt(252) * 100
        ax6.scatter(portfolio_annual_vol, portfolio_annual_ret, 
                   s=200, color='red', marker='*', label='Portfolio', zorder=5)
        
        ax6.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Volatility (%)')
        ax6.set_ylabel('Annual Return (%)')
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self._print(f"Saved plot to {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> None:
        """Generate comprehensive backtest report"""
        print("\n" + "="*60)
        print(" PORTFOLIO BACKTEST REPORT")
        print("="*60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        if self.portfolio_value is not None and len(self.portfolio_value) > 0:
            print(f"Final Value: ${self.portfolio_value.iloc[-1]:,.2f}")
        
        print("\n" + "="*60)
        print(" PERFORMANCE METRICS")
        print("="*60)
        
        for metric, value in self.metrics.items():
            print(f"{metric:.<30} {value}")
        
        print("\n" + "="*60)
        print(" PORTFOLIO COMPOSITION")
        print("="*60)
        
        for ticker, info in sorted(self.portfolio.items(), 
                                  key=lambda x: x[1]['normalized_weight'], 
                                  reverse=True):
            print(f"{info['name']:.<25} {info['normalized_weight']*100:>6.2f}% ({info['type']})")
        
        print("\n" + "="*60)
        print(" INDIVIDUAL ASSET PERFORMANCE")
        print("="*60)
        
        asset_performance = {}
        for ticker in self.price_data:
            if len(self.price_data[ticker]) > 1:
                total_return = (self.price_data[ticker].iloc[-1] / 
                              self.price_data[ticker].iloc[0] - 1) * 100
                asset_performance[ticker] = total_return
        
        for ticker, ret in sorted(asset_performance.items(), 
                                 key=lambda x: x[1], reverse=True):
            if ticker in self.portfolio:
                print(f"{self.portfolio[ticker]['name']:.<25} {ret:>7.2f}%")
        
        # Year-by-year breakdown
        print("\n" + "="*60)
        print(" YEAR-BY-YEAR PERFORMANCE")
        print("="*60)
        
        try:
            yearly_returns = self.portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
            for date, ret in yearly_returns.items():
                print(f"{date.year:.<25} {ret:>7.2f}%")
        except:
            print("Insufficient data for yearly breakdown")
    
    def export_results(self, filename: str) -> None:
        """Export results to CSV file"""
        results_df = pd.DataFrame({
            'Date': self.portfolio_value.index,
            'Portfolio Value': self.portfolio_value.values,
            'Daily Return': self.portfolio_returns.values
        })
        results_df.to_csv(filename, index=False)
        self._print(f"Results exported to {filename}")
    
    def run(self) -> None:
        """Run the complete backtest"""
        self.fetch_data()
        self.calculate_returns()
        self.calculate_portfolio_performance()
        self.calculate_metrics()