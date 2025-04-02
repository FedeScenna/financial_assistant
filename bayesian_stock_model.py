"""
Bayesian Stock Returns Prediction Model

This script creates a Bayesian model to predict next-day stock returns using historical price data.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the price data.
    
    Args:
        file_path (str): Path to the price data CSV file
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed features and target variables
    """
    logger.info("Loading price data...")
    df = pd.read_csv(file_path)
    
    # Calculate daily returns
    df['returns'] = df.groupby('ticker')['close'].pct_change()
    
    # Calculate rolling statistics
    window = 20  # 20-day window
    
    # Volatility (standard deviation of returns)
    df['volatility'] = df.groupby('ticker')['returns'].rolling(window=window).std().reset_index(0, drop=True)
    
    # Rolling mean returns
    df['rolling_mean'] = df.groupby('ticker')['returns'].rolling(window=window).mean().reset_index(0, drop=True)
    
    # Price momentum (20-day)
    df['momentum'] = df.groupby('ticker')['close'].pct_change(window).reset_index(0, drop=True)
    
    # Volume momentum
    df['volume_momentum'] = df.groupby('ticker')['volume'].pct_change(window).reset_index(0, drop=True)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    # Prepare features and target
    features = ['volatility', 'rolling_mean', 'momentum', 'volume_momentum']
    target = 'returns'
    
    X = df[features]
    y = df[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

def create_bayesian_model(X: pd.DataFrame, y: pd.Series) -> pm.Model:
    """
    Create a Bayesian linear regression model.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        
    Returns:
        pm.Model: PyMC model
    """
    logger.info("Creating Bayesian model...")
    
    with pm.Model() as model:
        # Priors for regression coefficients
        beta = pm.Normal('beta', mu=0, sigma=10, shape=len(X.columns))
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        
        # Linear regression
        mu = alpha + pm.math.dot(X, beta)
        
        # Likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=y)
    
    return model

def train_model(model: pm.Model, X: pd.DataFrame, y: pd.Series, 
                draws: int = 2000, tune: int = 1000, chains: int = 4) -> az.InferenceData:
    """
    Train the Bayesian model using MCMC sampling.
    
    Args:
        model (pm.Model): PyMC model
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        draws (int): Number of posterior samples
        tune (int): Number of tuning samples
        chains (int): Number of MCMC chains
        
    Returns:
        az.InferenceData: ArviZ inference data object
    """
    logger.info("Training Bayesian model...")
    
    with model:
        # Use NUTS sampler with multiple chains
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=4,
            return_inferencedata=True,
            random_seed=42
        )
    
    return idata

def plot_posterior_distributions(idata: az.InferenceData, output_dir: str):
    """
    Plot posterior distributions of model parameters.
    
    Args:
        idata (az.InferenceData): ArviZ inference data
        output_dir (str): Directory to save plots
    """
    logger.info("Plotting posterior distributions...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot posterior distributions
    az.plot_posterior(idata, var_names=['beta', 'alpha', 'sigma'])
    plt.savefig(Path(output_dir) / 'posterior_distributions.png')
    plt.close()
    
    # Plot trace
    az.plot_trace(idata)
    plt.savefig(Path(output_dir) / 'trace_plot.png')
    plt.close()
    
    # Plot energy distribution
    az.plot_energy(idata)
    plt.savefig(Path(output_dir) / 'energy_distribution.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define paths
    data_path = "data/price_data.csv"
    output_dir = "models/bayesian"
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(data_path)
    
    # Create and train model
    model = create_bayesian_model(X, y)
    idata = train_model(model, X, y)
    
    # Plot results
    plot_posterior_distributions(idata, output_dir)
    
    # Save model summary
    summary = az.summary(idata, var_names=['beta', 'alpha', 'sigma'])
    summary.to_csv(Path(output_dir) / 'model_summary.csv')
    
    # Save model diagnostics
    diagnostics = az.diagnostics(idata)
    diagnostics.to_csv(Path(output_dir) / 'model_diagnostics.csv')
    
    logger.info("Model training completed. Results saved to %s", output_dir)

if __name__ == "__main__":
    main() 