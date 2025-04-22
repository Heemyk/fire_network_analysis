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

logger.info("Starting connectivity analysis...")
logger.info("Loading graph from file...")
G = nx.read_gml(os.path.join(root_dir, "data", "fire_percolation_graph.gml"))
logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Analyze connected components
logger.info("Analyzing connected components...")
components = list(nx.connected_components(G))
component_sizes = [len(c) for c in components]
largest_component = max(components, key=len)

logger.info(f"Number of connected components: {len(components)}")
logger.info(f"Size of largest component: {len(largest_component)}")
logger.info(f"Percentage of nodes in largest component: {len(largest_component)/G.number_of_nodes()*100:.2f}%")

# Plot component size distribution
plt.figure(figsize=(10, 6))
plt.hist(component_sizes, bins=50, log=True)
plt.xlabel("Component Size")
plt.ylabel("Frequency (log scale)")
plt.title("Distribution of Connected Component Sizes")
plt.savefig(os.path.join(root_dir, "data", "component_size_distribution.png"))
logger.info("Saved component size distribution plot")

# Analyze node degrees
degrees = [d for n, d in G.degree()]
logger.info(f"Average node degree: {np.mean(degrees):.2f}")
logger.info(f"Median node degree: {np.median(degrees):.2f}")
logger.info(f"Min node degree: {min(degrees)}")
logger.info(f"Max node degree: {max(degrees)}")

# Plot degree distribution
plt.figure(figsize=(10, 6))
plt.hist(degrees, bins=50, log=True)
plt.xlabel("Node Degree")
plt.ylabel("Frequency (log scale)")
plt.title("Distribution of Node Degrees")
plt.savefig(os.path.join(root_dir, "data", "degree_distribution.png"))
logger.info("Saved degree distribution plot")

# Analyze edge probabilities
edge_probs = [data['p'] for u, v, data in G.edges(data=True)]
logger.info(f"Edge probability statistics:")
logger.info(f"Min probability: {min(edge_probs):.4f}")
logger.info(f"Max probability: {max(edge_probs):.4f}")
logger.info(f"Mean probability: {np.mean(edge_probs):.4f}")
logger.info(f"Median probability: {np.median(edge_probs):.4f}")

# Plot edge probability distribution
plt.figure(figsize=(10, 6))
plt.hist(edge_probs, bins=50)
plt.xlabel("Edge Probability")
plt.ylabel("Frequency")
plt.title("Distribution of Edge Probabilities")
plt.savefig(os.path.join(root_dir, "data", "edge_probability_distribution.png"))
logger.info("Saved edge probability distribution plot")

logger.info("Connectivity analysis complete!") 