import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

def analyze_distributions():
    """Analyze the distributions of fire probabilities and vegetation."""
    logger.info("Loading fire data...")
    df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Fire probability distribution
    ax1 = plt.subplot(221)
    sns.histplot(data=df, x='MEAN_FIRE', bins=50, ax=ax1)
    ax1.set_title('Distribution of Fire Probabilities')
    ax1.set_xlabel('Fire Probability')
    
    # 2. Vegetation distribution
    ax2 = plt.subplot(222)
    sns.histplot(data=df, x='MEAN_VEG', bins=50, ax=ax2)
    ax2.set_title('Distribution of Vegetation')
    ax2.set_xlabel('Mean Vegetation')
    
    # 3. Fire probability vs Vegetation scatter with density
    ax3 = plt.subplot(223)
    sns.kdeplot(data=df, x='MEAN_VEG', y='MEAN_FIRE', ax=ax3)
    ax3.set_title('Fire Probability vs Vegetation (Density)')
    
    # 4. Box plot of fire probabilities
    ax4 = plt.subplot(224)
    sns.boxplot(data=df, y='MEAN_FIRE', ax=ax4)
    ax4.set_title('Box Plot of Fire Probabilities')
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "data", "data_distribution_analysis.png"))
    
    # Print summary statistics
    logger.info("\nFire Probability Statistics:")
    logger.info(df['MEAN_FIRE'].describe())
    
    # Check for unique values
    unique_fire_probs = sorted(df['MEAN_FIRE'].unique())
    logger.info(f"\nNumber of unique fire probability values: {len(unique_fire_probs)}")
    logger.info("First few unique values:")
    logger.info(unique_fire_probs[:10])
    
    # Calculate proportion of zeros and high values
    n_zeros = (df['MEAN_FIRE'] < 1).sum()
    n_high = (df['MEAN_FIRE'] > 200).sum()
    total = len(df)
    
    logger.info(f"\nProportion of near-zero values (<1): {n_zeros/total:.2%}")
    logger.info(f"Proportion of high values (>200): {n_high/total:.2%}")

if __name__ == "__main__":
    analyze_distributions() 