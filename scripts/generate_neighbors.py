import pandas as pd
import numpy as np
import os
import logging
import re
from string import ascii_uppercase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

def parse_hex_id(hex_id):
    """Parse hexagon ID into its components (column and row)."""
    # Format is like "RS-919" or "A-178"
    match = re.match(r'([A-Z]+)-(\d+)', str(hex_id))
    if match:
        column, row = match.groups()
        return column, int(row)
    return None, None

def get_next_column(column):
    """Get the next column in the sequence."""
    if len(column) == 1:
        # Single letter columns (A, B, C, ...)
        if column == 'Z':
            return 'AA'
        return chr(ord(column) + 1)
    else:
        # Double letter columns (AA, AB, AC, ...)
        first, second = column[0], column[1]
        if second == 'Z':
            return chr(ord(first) + 1) + 'A'
        return first + chr(ord(second) + 1)

def get_prev_column(column):
    """Get the previous column in the sequence."""
    if len(column) == 1:
        # Single letter columns (A, B, C, ...)
        if column == 'A':
            return None
        return chr(ord(column) - 1)
    else:
        # Double letter columns (AA, AB, AC, ...)
        first, second = column[0], column[1]
        if second == 'A':
            if first == 'A':
                return 'Z'
            return chr(ord(first) - 1) + 'Z'
        return first + chr(ord(second) - 1)

def get_hex_neighbors(column, row):
    """Get the 6 neighboring hexagon coordinates."""
    # For a hex grid with this ID system, neighbors are:
    # 1. Same column, row + 1
    # 2. Same column, row - 1
    # 3. Next column, same row
    # 4. Next column, row - 1
    # 5. Previous column, same row
    # 6. Previous column, row + 1
    neighbors = []
    
    # Same column neighbors
    neighbors.append((column, row + 1))
    neighbors.append((column, row - 1))
    
    # Next column neighbors
    next_col = get_next_column(column)
    if next_col:
        neighbors.append((next_col, row))
        neighbors.append((next_col, row - 1))
    
    # Previous column neighbors
    prev_col = get_prev_column(column)
    if prev_col:
        neighbors.append((prev_col, row))
        neighbors.append((prev_col, row + 1))
    
    return neighbors

def generate_neighbors(hex_ids):
    """Generate neighbor relationships for a list of hexagon IDs."""
    # Create a mapping from coordinates to hex IDs
    coord_to_id = {}
    id_to_coord = {}
    
    for hex_id in hex_ids:
        column, row = parse_hex_id(hex_id)
        if column is not None and row is not None:
            coord_to_id[(column, row)] = hex_id
            id_to_coord[hex_id] = (column, row)
    
    # Generate neighbor relationships
    neighbors = []
    for hex_id, (column, row) in id_to_coord.items():
        for neighbor_col, neighbor_row in get_hex_neighbors(column, row):
            neighbor_id = coord_to_id.get((neighbor_col, neighbor_row))
            if neighbor_id is not None:
                neighbors.append((hex_id, neighbor_id))
    
    return neighbors

logger.info("Starting neighbor generation process...")

# Load fire data
logger.info("Loading fire data...")
fire_df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
hex_ids = fire_df['GRID_ID'].unique()
logger.info(f"Found {len(hex_ids)} unique hexagon IDs")

# Generate neighbor relationships
logger.info("Generating neighbor relationships...")
neighbors = generate_neighbors(hex_ids)
logger.info(f"Generated {len(neighbors)} neighbor relationships")

# Create DataFrame
neighbors_df = pd.DataFrame(neighbors, columns=['GRID_ID', 'GRID_ID_1'])

# Remove any potential duplicates
neighbors_df = neighbors_df.drop_duplicates()

# Remove self-loops (if any)
neighbors_df = neighbors_df[neighbors_df['GRID_ID'] != neighbors_df['GRID_ID_1']]

logger.info(f"Final number of unique neighbor relationships: {len(neighbors_df)}")

# Save to CSV
output_path = os.path.join(root_dir, "data", "hex_neighbors.csv")
neighbors_df.to_csv(output_path, index=False)
logger.info(f"Saved neighbor relationships to {output_path}")

# Verify the results
logger.info("Verifying results...")
total_nodes = len(hex_ids)
connected_nodes = len(set(neighbors_df['GRID_ID'].unique()) | set(neighbors_df['GRID_ID_1'].unique()))
logger.info(f"Total nodes: {total_nodes}")
logger.info(f"Connected nodes: {connected_nodes}")
logger.info(f"Percentage connected: {connected_nodes/total_nodes*100:.2f}%") 