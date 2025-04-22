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

logger.info("Starting data analysis...")

# Load the data files
logger.info("Loading fire data...")
fire_df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
logger.info("Loading neighbors data...")
neighbors_df = pd.read_csv(os.path.join(root_dir, "data", "neighbours_data.csv"))

# Get unique IDs from both files
fire_ids = set(fire_df['GRID_ID'].unique())
neighbor_ids = set(neighbors_df['GRID_ID'].unique()) | set(neighbors_df['GRID_ID_1'].unique())

logger.info(f"Number of unique IDs in fire data: {len(fire_ids)}")
logger.info(f"Number of unique IDs in neighbors data: {len(neighbor_ids)}")

# Find IDs that are in neighbors but not in fire data
missing_in_fire = neighbor_ids - fire_ids
logger.info(f"Number of IDs in neighbors but missing in fire data: {len(missing_in_fire)}")

# Find IDs that are in fire but not in neighbors
missing_in_neighbors = fire_ids - neighbor_ids
logger.info(f"Number of IDs in fire but missing in neighbors: {len(missing_in_neighbors)}")

# Sample some of the missing IDs
if missing_in_fire:
    logger.info("Sample of IDs missing in fire data:")
    logger.info(list(missing_in_fire)[:5])

if missing_in_neighbors:
    logger.info("Sample of IDs missing in neighbors:")
    logger.info(list(missing_in_neighbors)[:5])

# Check for duplicate edges
edge_pairs = set(zip(neighbors_df['GRID_ID'], neighbors_df['GRID_ID_1']))
logger.info(f"Number of unique edges in neighbors data: {len(edge_pairs)}")
logger.info(f"Number of duplicate edges: {len(neighbors_df) - len(edge_pairs)}")

# Check for self-loops
self_loops = neighbors_df[neighbors_df['GRID_ID'] == neighbors_df['GRID_ID_1']]
logger.info(f"Number of self-loops: {len(self_loops)}") 