#!/usr/bin/env python3
"""
Portfolio Backtester - Main Entry Point
"""

import argparse
import json
import sys
from datetime import datetime
from portfolio_backtest import PortfolioBacktester

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='Backtest investment portfolios with performance metrics and visualizations'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file (default: config.json)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip generating visualization plots'
    )
    parser.add_argument(
        '--export',
        type=str,
        help='Export results to CSV file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract settings
    settings = config.get('settings', {})
    start_date = settings.get('start_date', '2020-01-01')
    end_date = settings.get('end_date', 'auto')
    if end_date == 'auto':
        end_date = datetime.now().strftime('%Y-%m-%d')
    initial_capital = settings.get('initial_capital', 100000)
    
    # Extract portfolio
    portfolio = config.get('portfolio', {})
    
    if not portfolio:
        print("Error: No portfolio defined in configuration file.")
        sys.exit(1)
    
    # Create and run backtester
    print("\n" + "="*60)
    print("  PORTFOLIO BACKTESTER")
    print("="*60)
    print(f"Configuration: {args.config}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("="*60 + "\n")
    
    backtester = PortfolioBacktester(
        portfolio_dict=portfolio,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        verbose=not args.quiet
    )
    
    try:
        # Run the backtest
        backtester.run()
        
        # Generate report
        if not args.quiet:
            backtester.generate_report()
        
        # Plot results
        if not args.no_plot:
            print("\nGenerating visualizations...")
            backtester.plot_performance(save_path='portfolio_analysis.png')
            print("✓ Saved visualizations to 'portfolio_analysis.png'")
        
        # Export results if requested
        if args.export:
            backtester.export_results(args.export)
            print(f"✓ Exported results to '{args.export}'")
        
        print("\n✅ Backtest completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify ticker symbols are correct")
        print("3. Ensure date range is valid")
        print("4. Update yfinance: pip install --upgrade yfinance")
        sys.exit(1)

if __name__ == "__main__":
    main()