"""
Stock Analyzer

This module provides functionality for analyzing stock price data.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import numpy as np
import streamlit as st


class StockDataHandler:
    """
    Handler for stock price data analysis.
    
    This class provides methods for loading, analyzing, and visualizing stock price data.
    """
    
    def __init__(self, csv_path):
        """
        Initialize the StockDataHandler with the path to the CSV file.
        
        Args:
            csv_path (str): Path to the CSV file containing stock price data
        """
        self.csv_path = csv_path
        self._data = None
        self._tickers = None
    
    @property
    def data(self):
        """
        Lazy load the data to avoid loading it until needed.
        
        Returns:
            pandas.DataFrame: The loaded stock price data
        """
        if self._data is None:
            try:
                # Check if we can use a cached version
                cache_file = self.csv_path.replace('.csv', '_cache.pkl')
                if os.path.exists(cache_file) and os.path.getmtime(cache_file) > os.path.getmtime(self.csv_path):
                    self._data = pd.read_pickle(cache_file)
                else:
                    # Load the CSV file
                    self._data = pd.read_csv(self.csv_path)
                    # Convert date to datetime
                    self._data['date'] = pd.to_datetime(self._data['date'])
                    # Save to cache
                    self._data.to_pickle(cache_file)
            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")
                self._data = pd.DataFrame()
        return self._data
    
    @property
    def tickers(self):
        """
        Get the list of available tickers.
        
        Returns:
            list: List of ticker symbols
        """
        if self._tickers is None:
            self._tickers = sorted(self.data['ticker'].unique())
        return self._tickers
    
    def get_ticker_data(self, ticker):
        """
        Get data for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            pandas.DataFrame: Data for the specified ticker
        """
        if ticker not in self.tickers:
            return pd.DataFrame()
        return self.data[self.data['ticker'] == ticker].sort_values('date')
    
    def get_price_history(self, ticker, start_date=None, end_date=None):
        """
        Get price history for a specific ticker within a date range.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            pandas.DataFrame: Price history for the specified ticker and date range
        """
        df = self.get_ticker_data(ticker)
        if df.empty:
            return pd.DataFrame()
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        return df
    
    def plot_price_history(self, ticker, start_date=None, end_date=None, metric='close'):
        """
        Plot price history for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            metric (str, optional): Price metric to plot (close, open, high, low)
            
        Returns:
            io.BytesIO: Buffer containing the plot image
        """
        df = self.get_price_history(ticker, start_date, end_date)
        if df.empty:
            return None
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['date'], df[metric])
        plt.title(f"{ticker} {metric.capitalize()} Price")
        plt.xlabel("Date")
        plt.ylabel(f"{metric.capitalize()} Price ($)")
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
    
    def calculate_returns(self, ticker, start_date=None, end_date=None, period='1d'):
        """
        Calculate returns for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            period (str, optional): Return period (1d, 2d, 3d, 6d, 9d, 12d)
            
        Returns:
            dict: Dictionary containing return metrics
        """
        df = self.get_price_history(ticker, start_date, end_date)
        if df.empty:
            return None
        
        # Map period to column
        period_map = {
            '1d': 'return_1d',
            '2d': 'return_2d',
            '3d': 'return_3d',
            '6d': 'return_6d',
            '9d': 'return_9d',
            '12d': 'return_12d'
        }
        
        if period not in period_map:
            return None
        
        return_col = period_map[period]
        avg_return = df[return_col].mean()
        std_return = df[return_col].std()
        
        return {
            'ticker': ticker,
            'period': period,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': avg_return / std_return if std_return > 0 else 0
        }
    
    def compare_tickers(self, tickers, start_date=None, end_date=None, metric='close'):
        """
        Compare multiple tickers based on a metric.
        
        Args:
            tickers (list): List of ticker symbols
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            metric (str, optional): Price metric to compare (close, open, high, low)
            
        Returns:
            io.BytesIO: Buffer containing the comparison plot image
        """
        if not tickers:
            return None
        
        plt.figure(figsize=(12, 8))
        
        for ticker in tickers:
            df = self.get_price_history(ticker, start_date, end_date)
            if not df.empty:
                # Normalize to starting value for fair comparison
                normalized = df[metric] / df[metric].iloc[0]
                plt.plot(df['date'], normalized, label=ticker)
        
        plt.title(f"Normalized {metric.capitalize()} Price Comparison")
        plt.xlabel("Date")
        plt.ylabel(f"Normalized {metric.capitalize()} Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
    
    def get_latest_price(self, ticker):
        """
        Get the latest price for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            
        Returns:
            dict: Dictionary containing latest price information
        """
        df = self.get_ticker_data(ticker)
        if df.empty:
            return None
        
        latest = df.sort_values('date').iloc[-1]
        return {
            'ticker': ticker,
            'date': latest['date'],
            'close': latest['close'],
            'open': latest['open'],
            'high': latest['high'],
            'low': latest['low'],
            'volume': latest['volume']
        }
    
    def get_technical_indicators(self, ticker, start_date=None, end_date=None):
        """
        Get technical indicators for a specific ticker.
        
        Args:
            ticker (str): Ticker symbol
            start_date (str, optional): Start date in YYYY-MM-DD format
            end_date (str, optional): End date in YYYY-MM-DD format
            
        Returns:
            dict: Dictionary containing technical indicators
        """
        df = self.get_price_history(ticker, start_date, end_date)
        if df.empty:
            return None
        
        # Get the latest values
        latest = df.iloc[-1]
        
        return {
            'ticker': ticker,
            'date': latest['date'],
            'rsi': latest['rsi'],
            'macd': latest['macd'],
            'bb_low': latest['bb_low'],
            'bb_mid': latest['bb_mid'],
            'bb_high': latest['bb_high'],
            'volatility': latest['garman_klass_vol'],
            'atr': latest['atr']
        }
    
    def search_by_performance(self, top_n=10, period='1d', ascending=False):
        """
        Search for top/bottom performing stocks in a given period.
        
        Args:
            top_n (int, optional): Number of stocks to return
            period (str, optional): Return period (1d, 2d, 3d, 6d, 9d, 12d)
            ascending (bool, optional): Sort in ascending order if True
            
        Returns:
            pandas.DataFrame: Top/bottom performing stocks
        """
        period_map = {
            '1d': 'return_1d',
            '2d': 'return_2d',
            '3d': 'return_3d',
            '6d': 'return_6d',
            '9d': 'return_9d',
            '12d': 'return_12d'
        }
        
        if period not in period_map:
            return pd.DataFrame()
        
        return_col = period_map[period]
        
        # Get the latest date for each ticker
        latest_dates = self.data.groupby('ticker')['date'].max().reset_index()
        
        # Merge with the original data to get the latest records
        latest_data = pd.merge(self.data, latest_dates, on=['ticker', 'date'])
        
        # Sort by the return column
        sorted_data = latest_data.sort_values(return_col, ascending=ascending)
        
        # Select top N
        result = sorted_data.head(top_n)[['ticker', 'date', return_col, 'close']]
        result.columns = ['Ticker', 'Date', f'Return ({period})', 'Close Price']
        
        return result


def extract_ticker_from_query(query):
    """
    Extract ticker symbol from a query.
    
    Args:
        query (str): The query text
        
    Returns:
        str or None: Extracted ticker symbol or None if not found
    """
    # This is a simple implementation - could be enhanced with NLP
    query = query.upper()
    words = query.split()
    
    # Check for common patterns like "AAPL stock" or "stock AAPL"
    for i, word in enumerate(words):
        # Remove any punctuation
        clean_word = ''.join(c for c in word if c.isalnum())
        if clean_word.isalpha() and len(clean_word) <= 5 and clean_word.isupper():
            return clean_word
    
    return None


def extract_date_range_from_query(query):
    """
    Extract date range from a query.
    
    Args:
        query (str): The query text
        
    Returns:
        tuple: (start_date, end_date) in YYYY-MM-DD format
    """
    # This is a simple implementation - could be enhanced with NLP
    today = datetime.now()
    
    # Check for common patterns
    if "last week" in query.lower():
        end_date = today
        start_date = today - timedelta(days=7)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    if "last month" in query.lower():
        end_date = today
        start_date = today - timedelta(days=30)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    if "last year" in query.lower():
        end_date = today
        start_date = today - timedelta(days=365)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    if "ytd" in query.lower() or "year to date" in query.lower():
        end_date = today
        start_date = datetime(today.year, 1, 1)
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    # Default to last 3 months
    end_date = today
    start_date = today - timedelta(days=90)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def process_stock_query(query, stock_handler):
    """
    Process a stock-related query and return relevant information.
    
    Args:
        query (str): The query text
        stock_handler (StockDataHandler): Instance of StockDataHandler
        
    Returns:
        str: Response to the query
    """
    query_lower = query.lower()
    
    # Extract ticker
    ticker = extract_ticker_from_query(query)
    if not ticker:
        return "I couldn't identify a specific stock ticker in your query. Please specify a stock symbol like AAPL, MSFT, etc."
    
    # Extract date range
    start_date, end_date = extract_date_range_from_query(query)
    
    # Determine query type
    if "price" in query_lower or "stock price" in query_lower or "how much" in query_lower:
        # Price query
        latest = stock_handler.get_latest_price(ticker)
        if not latest:
            return f"I couldn't find price data for {ticker}."
        
        return f"The latest price for {ticker} as of {latest['date'].strftime('%Y-%m-%d')} is:\n" \
               f"Close: ${latest['close']:.2f}\n" \
               f"Open: ${latest['open']:.2f}\n" \
               f"High: ${latest['high']:.2f}\n" \
               f"Low: ${latest['low']:.2f}\n" \
               f"Volume: {int(latest['volume']):,}"
    
    elif "technical" in query_lower or "indicator" in query_lower or "rsi" in query_lower or "macd" in query_lower:
        # Technical indicators query
        indicators = stock_handler.get_technical_indicators(ticker)
        if not indicators:
            return f"I couldn't find technical indicator data for {ticker}."
        
        return f"Technical indicators for {ticker} as of {indicators['date'].strftime('%Y-%m-%d')}:\n" \
               f"RSI: {indicators['rsi']:.2f}\n" \
               f"MACD: {indicators['macd']:.2f}\n" \
               f"Bollinger Bands: Low=${indicators['bb_low']:.2f}, Mid=${indicators['bb_mid']:.2f}, High=${indicators['bb_high']:.2f}\n" \
               f"Volatility: {indicators['volatility']:.4f}\n" \
               f"ATR: {indicators['atr']:.4f}"
    
    elif "return" in query_lower or "performance" in query_lower:
        # Return/performance query
        period = '1d'  # default
        if "week" in query_lower:
            period = '6d'
        elif "month" in query_lower:
            period = '12d'
        
        returns = stock_handler.calculate_returns(ticker, start_date, end_date, period)
        if not returns:
            return f"I couldn't find return data for {ticker}."
        
        return f"Performance metrics for {ticker} ({period} returns):\n" \
               f"Average Return: {returns['avg_return']*100:.2f}%\n" \
               f"Standard Deviation: {returns['std_return']*100:.2f}%\n" \
               f"Sharpe Ratio: {returns['sharpe_ratio']:.2f}"
    
    elif "compare" in query_lower and "to" in query_lower:
        # Comparison query - try to extract second ticker
        words = query.split()
        second_ticker = None
        for i, word in enumerate(words):
            if word.lower() == "to" and i+1 < len(words):
                potential_ticker = words[i+1].upper().strip(".,;:")
                if potential_ticker.isalpha() and len(potential_ticker) <= 5:
                    second_ticker = potential_ticker
                    break
        
        if not second_ticker:
            return f"I understood you want to compare {ticker} to another stock, but couldn't identify the second ticker."
        
        # Generate comparison text
        ticker_data = stock_handler.get_price_history(ticker, start_date, end_date)
        second_data = stock_handler.get_price_history(second_ticker, start_date, end_date)
        
        if ticker_data.empty or second_data.empty:
            return f"I couldn't find sufficient data to compare {ticker} and {second_ticker}."
        
        ticker_return = ticker_data['return_1d'].mean() * 100
        second_return = second_data['return_1d'].mean() * 100
        
        return f"Comparison between {ticker} and {second_ticker} from {start_date} to {end_date}:\n" \
               f"{ticker} average daily return: {ticker_return:.2f}%\n" \
               f"{second_ticker} average daily return: {second_return:.2f}%\n" \
               f"Relative performance: {ticker} {'outperformed' if ticker_return > second_return else 'underperformed'} {second_ticker} by {abs(ticker_return - second_return):.2f}%"
    
    else:
        # General query - return latest price and basic info
        latest = stock_handler.get_latest_price(ticker)
        if not latest:
            return f"I couldn't find data for {ticker}."
        
        return f"Information for {ticker} as of {latest['date'].strftime('%Y-%m-%d')}:\n" \
               f"Latest price: ${latest['close']:.2f}\n" \
               f"Daily range: ${latest['low']:.2f} - ${latest['high']:.2f}\n" \
               f"Volume: {int(latest['volume']):,}"


def load_stock_data(csv_path):
    """
    Load stock data from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        StockDataHandler: Instance of StockDataHandler
    """
    try:
        stock_handler = StockDataHandler(csv_path)
        return stock_handler
    except Exception as e:
        st.error(f"Error loading stock data: {str(e)}")
        return None 