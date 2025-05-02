"""
Bayesian Stock Return Prediction Model

This module implements a Bayesian model for predicting stock returns using PyMC.
The model supports GPU acceleration for faster inference.
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import torch
from typing import Tuple, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
import time
from pytensor.config import config

class BayesianStockModel:
    def __init__(self, use_gpu: bool = True):
        """
        Initialize the Bayesian Stock Model.
        
        Args:
            use_gpu (bool): Whether to use GPU for computation if available
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.model = None
        self.trace = None
        self.data = None
        
        # Configure PyMC to use GPU if available
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            config.floatX = "float32"  # Use float32 for GPU compatibility
            config.device = "cuda"
        else:
            print("Using CPU for computation")
            config.device = "cpu"
    
    def load_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """
        Load and preprocess stock price data for multiple tickers.
        
        Args:
            data_path (str): Path to the CSV file containing price data
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of preprocessed stock data by ticker
        """
        print("Loading and preprocessing data...")
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Sort by ticker and date
        df = df.sort_values(['ticker', 'date'])
        
        # Dictionary to store data by ticker
        ticker_data = {}
        
        # Process each ticker separately
        for ticker in df['ticker'].unique():
            print(f"\nProcessing {ticker}...")
            
            # Get data for this ticker
            ticker_df = df[df['ticker'] == ticker].copy()
            
            # Calculate daily returns
            ticker_df['returns'] = ticker_df['close'].pct_change()
            
            # Calculate additional features
            ticker_df['log_returns'] = np.log1p(ticker_df['returns'])
            ticker_df['volatility'] = ticker_df['returns'].rolling(window=20).std()
            ticker_df['volume_change'] = ticker_df['volume'].pct_change()
            
            # Remove NaN values
            ticker_df = ticker_df.dropna()
            
            # Store in dictionary
            ticker_data[ticker] = ticker_df
            
            print(f"Processed {len(ticker_df)} days of data for {ticker}")
            print(f"Date range: {ticker_df['date'].min()} to {ticker_df['date'].max()}")
            print(f"Mean return: {ticker_df['returns'].mean():.4f}")
            print(f"Return volatility: {ticker_df['returns'].std():.4f}")
        
        self.data = ticker_data
        return ticker_data
    
    def build_model(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pm.Model]:
        """
        Build the Bayesian model for stock returns for each ticker.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary of preprocessed stock data by ticker
            
        Returns:
            Dict[str, pm.Model]: Dictionary of PyMC models by ticker
        """
        print("Building Bayesian models...")
        
        models = {}
        for ticker, ticker_data in data.items():
            print(f"\nBuilding model for {ticker}...")
            
            with pm.Model() as model:
                # Priors
                mu = pm.Normal('mu', mu=0, sigma=0.1)
                sigma = pm.HalfNormal('sigma', sigma=0.1)
                
                # GARCH parameters
                alpha = pm.Beta('alpha', alpha=1, beta=1)
                beta = pm.Beta('beta', alpha=1, beta=1)
                
                # Initialize volatility process
                volatility = pm.Deterministic(
                    'volatility',
                    pt.sqrt(sigma**2 + alpha * pt.square(ticker_data['returns'].shift(1).fillna(0)))
                )
                
                # Update volatility with GARCH effect
                for t in range(1, len(ticker_data)):
                    volatility = pt.set_subtensor(
                        volatility[t],
                        pt.sqrt(
                            sigma**2 + 
                            alpha * pt.square(ticker_data['returns'].iloc[t-1]) + 
                            beta * pt.square(volatility[t-1])
                        )
                    )
                
                # Likelihood
                returns = pm.Normal('returns', mu=mu, sigma=volatility, observed=ticker_data['returns'])
            
            models[ticker] = model
        
        self.model = models
        return models
    
    def train(self, 
              data: Dict[str, pd.DataFrame], 
              draws: int = 2000, 
              tune: int = 1000, 
              chains: int = 4) -> Dict[str, az.InferenceData]:
        """
        Train the Bayesian model for each ticker.
        
        Args:
            data (Dict[str, pd.DataFrame]): Dictionary of preprocessed stock data by ticker
            draws (int): Number of posterior samples
            tune (int): Number of tuning samples
            chains (int): Number of MCMC chains
            
        Returns:
            Dict[str, az.InferenceData]: Dictionary of inference data by ticker
        """
        if self.model is None:
            self.build_model(data)
        
        print("\n=== Training Configuration ===")
        print(f"Number of draws: {draws}")
        print(f"Tuning samples: {tune}")
        print(f"Number of chains: {chains}")
        print(f"Using GPU: {self.use_gpu}")
        print("===========================\n")
        
        traces = {}
        
        for ticker, model in self.model.items():
            print(f"\n=== Training Model for {ticker} ===")
            print(f"Data points: {len(data[ticker])}")
            print(f"Date range: {data[ticker]['date'].min()} to {data[ticker]['date'].max()}")
            print(f"Mean return: {data[ticker]['returns'].mean():.4f}")
            print(f"Return volatility: {data[ticker]['returns'].std():.4f}")
            
            with model:
                # Use NUTS sampler with GPU if available
                step = pm.NUTS(target_accept=0.9)
                
                print("\nStarting MCMC sampling...")
                start_time = time.time()
                
                # Run MCMC
                trace = pm.sample(
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    step=step,
                    cores=1,  # Use single core to avoid memory issues
                    progressbar=True,
                    return_inferencedata=True
                )
                
                end_time = time.time()
                print(f"\nSampling completed in {end_time - start_time:.2f} seconds")
                
                # Print model summary
                print("\nModel Summary:")
                print(az.summary(trace))
                
                # Print convergence diagnostics
                print("\nConvergence Diagnostics:")
                print(f"R-hat: {az.rhat(trace).max():.4f}")
                print(f"Effective sample size: {az.ess(trace).min():.0f}")
                
                # Print parameter estimates
                print("\nParameter Estimates:")
                for var in ['mu', 'sigma', 'alpha', 'beta']:
                    mean = float(trace.posterior[var].mean())
                    std = float(trace.posterior[var].std())
                    print(f"{var}: {mean:.4f} ± {std:.4f}")
            
            traces[ticker] = trace
            print(f"\n=== Completed Training for {ticker} ===\n")
        
        self.trace = traces
        return traces
    
    def predict(self, 
                days: int = 30, 
                samples: int = 1000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions for future returns for each ticker.
        
        Args:
            days (int): Number of days to predict
            samples (int): Number of samples to draw
            
        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary of predictions and confidence intervals by ticker
        """
        if self.trace is None:
            raise ValueError("Model must be trained before making predictions")
        
        print(f"Generating {days}-day predictions...")
        predictions = {}
        
        for ticker, trace in self.trace.items():
            print(f"\nGenerating predictions for {ticker}...")
            
            # Get posterior samples
            mu_samples = trace.posterior['mu'].values.flatten()
            sigma_samples = trace.posterior['sigma'].values.flatten()
            
            # Generate predictions
            ticker_predictions = np.zeros((samples, days))
            for i in range(samples):
                mu = np.random.choice(mu_samples)
                sigma = np.random.choice(sigma_samples)
                ticker_predictions[i] = np.random.normal(mu, sigma, days)
            
            # Calculate confidence intervals
            lower = np.percentile(ticker_predictions, 2.5, axis=0)
            upper = np.percentile(ticker_predictions, 97.5, axis=0)
            
            predictions[ticker] = (ticker_predictions, (lower, upper))
        
        return predictions
    
    def save_model(self, output_dir: str = "models") -> None:
        """
        Save the trained model and its parameters.
        
        Args:
            output_dir (str): Directory to save the model
        """
        if self.trace is None:
            raise ValueError("No trained model to save")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save model parameters
        params = {
            'mu_mean': float(self.trace.posterior['mu'].mean()),
            'mu_std': float(self.trace.posterior['mu'].std()),
            'sigma_mean': float(self.trace.posterior['sigma'].mean()),
            'sigma_std': float(self.trace.posterior['sigma'].std()),
            'training_date': datetime.now().isoformat()
        }
        
        with open(output_path / 'model_params.json', 'w') as f:
            json.dump(params, f, indent=4)
        
        # Save trace
        az.to_netcdf(self.trace, str(output_path / 'trace.nc'))
        
        print(f"Model saved to {output_path}")
    
    def load_model(self, model_dir: str = "models") -> None:
        """
        Load a trained model.
        
        Args:
            model_dir (str): Directory containing the saved model
        """
        model_path = Path(model_dir)
        
        if not (model_path / 'trace.nc').exists():
            raise FileNotFoundError(f"No saved model found in {model_path}")
        
        # Load trace
        self.trace = az.from_netcdf(str(model_path / 'trace.nc'))
        
        # Rebuild model
        self.build_model(self.data)
        
        print(f"Model loaded from {model_path}")

def main():
    """Example usage of the Bayesian Stock Model."""
    # Initialize model
    model = BayesianStockModel(use_gpu=True)
    
    # Load data
    data = model.load_data("data/price_data.csv")
    
    # Train model
    traces = model.train(data)
    
    # Generate predictions
    predictions = model.predict(days=30)
    
    # Save model
    model.save_model()
    
    # Print summary for each ticker
    for ticker, trace in traces.items():
        print(f"\nModel Summary for {ticker}:")
        print(az.summary(trace))
        
        pred, (lower, upper) = predictions[ticker]
        print(f"\nPrediction Summary for {ticker}:")
        print(f"Mean return: {pred.mean():.4f}")
        print(f"95% CI: [{lower.mean():.4f}, {upper.mean():.4f}]")

if __name__ == "__main__":
    main() 