import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

logger.info("Loading fire data to examine GRID_ID format...")
fire_df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))

# Display first 10 GRID_IDs
logger.info("First 10 GRID_IDs:")
for i, grid_id in enumerate(fire_df['GRID_ID'].head(10)):
    logger.info(f"{i+1}. {grid_id}")

# Check data type
logger.info(f"\nGRID_ID data type: {fire_df['GRID_ID'].dtype}")

# Check for any patterns in the IDs
logger.info("\nAnalyzing ID patterns...")
unique_lengths = fire_df['GRID_ID'].astype(str).str.len().unique()
logger.info(f"Unique ID lengths: {unique_lengths}")

# Check for any non-numeric characters
non_numeric = fire_df['GRID_ID'].astype(str).str.contains(r'[^0-9]').sum()
logger.info(f"Number of IDs with non-numeric characters: {non_numeric}")

# Display some statistics
logger.info("\nGRID_ID statistics:")
logger.info(f"Total number of IDs: {len(fire_df['GRID_ID'])}")
logger.info(f"Number of unique IDs: {fire_df['GRID_ID'].nunique()}")
logger.info(f"Minimum ID: {fire_df['GRID_ID'].min()}")
logger.info(f"Maximum ID: {fire_df['GRID_ID'].max()}") 