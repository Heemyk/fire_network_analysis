import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

logger.info("Starting data loading process...")

# Load hexagon data (includes fire probabilities, vegetation, etc.)
logger.info("Loading fire data from CSV...")
df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
logger.info(f"Loaded {len(df)} rows of fire data")

# Load predicted fire probabilities
logger.info("Loading predicted fire probabilities...")
pred_df = pd.read_csv(os.path.join(root_dir, "data", "predicted_fire_probs.csv"))
df = df.merge(pred_df, on='GRID_ID', how='left')
logger.info("Merged predictions with fire data")

# Load generated neighbors data
logger.info("Loading generated neighbors data from CSV...")
neighbors_df = pd.read_csv(os.path.join(root_dir, "data", "hex_neighbors.csv"))
logger.info(f"Loaded {len(neighbors_df)} neighbor relationships")

logger.info("Constructing graph...")
G = nx.Graph()

# Add nodes with attributes
logger.info("Adding nodes to graph...")
for _, row in df.iterrows():
    G.add_node(row['GRID_ID'], 
               actual_fire_prob=row['MEAN_FIRE']/100,
               predicted_fire_prob=row['predicted_fire_prob'],
               veg_fuel=row['MEAN_VEG'])
logger.info(f"Added {G.number_of_nodes()} nodes to graph")

# Add edges using the neighbors list
logger.info("Adding edges to graph...")
for _, row in neighbors_df.iterrows():
    node1 = row['GRID_ID']
    node2 = row['GRID_ID_1']
    G.add_edge(node1, node2)
logger.info(f"Added {G.number_of_edges()} edges to graph")

# Verify graph connectivity
components = list(nx.connected_components(G))
largest_component = max(components, key=len)
logger.info(f"Number of connected components: {len(components)}")
logger.info(f"Size of largest component: {len(largest_component)}")
logger.info(f"Percentage of nodes in largest component: {len(largest_component)/G.number_of_nodes()*100:.2f}%")

# Calculate edge probabilities based on predicted fire probabilities
logger.info("Calculating edge probabilities...")
all_probs = []
for u, v in G.edges:
    prob_u = G.nodes[u]['predicted_fire_prob']
    prob_v = G.nodes[v]['predicted_fire_prob']
    # Edge probability is average of node probabilities
    edge_prob = (prob_u + prob_v) / 2
    G.edges[u, v]['p'] = edge_prob
    all_probs.append(edge_prob)

# Log probability statistics
if all_probs:
    logger.info(f"Edge probability statistics:")
    logger.info(f"Min probability: {min(all_probs):.4f}")
    logger.info(f"Max probability: {max(all_probs):.4f}")
    logger.info(f"Mean probability: {np.mean(all_probs):.4f}")
    logger.info(f"Median probability: {np.median(all_probs):.4f}")

# Save as a .gpickle file (best for retaining all attributes)
logger.info("Saving graph to file...")
nx.write_gml(G, os.path.join(root_dir, "data", "fire_percolation_graph.gml"))
logger.info("Graph saved successfully!")

