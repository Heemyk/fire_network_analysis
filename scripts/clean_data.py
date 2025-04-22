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

logger.info("Starting data cleaning process...")

# Load the data files
logger.info("Loading fire data...")
fire_df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
logger.info("Loading neighbors data...")
neighbors_df = pd.read_csv(os.path.join(root_dir, "data", "neighbours_data.csv"))

# Get unique IDs from fire data
fire_ids = set(fire_df['GRID_ID'].unique())
logger.info(f"Number of unique IDs in fire data: {len(fire_ids)}")

# Clean neighbors data
logger.info("Cleaning neighbors data...")

# Remove self-loops
neighbors_df = neighbors_df[neighbors_df['GRID_ID'] != neighbors_df['GRID_ID_1']]
logger.info(f"Removed self-loops. Remaining rows: {len(neighbors_df)}")

# Remove duplicate edges
neighbors_df = neighbors_df.drop_duplicates(subset=['GRID_ID', 'GRID_ID_1'])
logger.info(f"Removed duplicate edges. Remaining rows: {len(neighbors_df)}")

# Keep only edges where both nodes exist in fire data
neighbors_df = neighbors_df[
    (neighbors_df['GRID_ID'].isin(fire_ids)) & 
    (neighbors_df['GRID_ID_1'].isin(fire_ids))
]
logger.info(f"Kept only edges with both nodes in fire data. Remaining rows: {len(neighbors_df)}")

# Save cleaned data
logger.info("Saving cleaned neighbors data...")
neighbors_df.to_csv(os.path.join(root_dir, "data", "neighbours_data_cleaned.csv"), index=False)

# Create a list of nodes that are in fire data but not in neighbors
isolated_nodes = fire_ids - set(neighbors_df['GRID_ID'].unique()) - set(neighbors_df['GRID_ID_1'].unique())
logger.info(f"Number of isolated nodes (no neighbors): {len(isolated_nodes)}")

# Save list of isolated nodes
pd.DataFrame({'GRID_ID': list(isolated_nodes)}).to_csv(
    os.path.join(root_dir, "data", "isolated_nodes.csv"), 
    index=False
)

logger.info("Data cleaning complete!") 