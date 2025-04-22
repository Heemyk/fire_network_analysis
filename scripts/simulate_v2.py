import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)

logger.info("Starting simulation process...")
logger.info("Loading graph from file...")
G = nx.read_gml(os.path.join(root_dir, "data", "fire_percolation_graph.gml"))
logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

def simulate_fire(G, p_threshold, prob_type='predicted'):
    """
    Simulate one run of fire spread at threshold p.
    prob_type: 'predicted' or 'actual' to use different probability types
    """
    G_sub = nx.Graph()
    G_sub.add_nodes_from(G.nodes(data=True))
    
    added_edges = 0
    total_edges = G.number_of_edges()
    
    for u, v, data in G.edges(data=True):
        # Get node probabilities
        if prob_type == 'predicted':
            edge_prob = G.nodes[u]['predicted_fire_prob']
        else:  # actual
            edge_prob = G.nodes[u]['actual_fire_prob']
        
        # Simplified edge formation - single probability check
        if random.random() < edge_prob and edge_prob >= p_threshold:
            G_sub.add_edge(u, v)
            added_edges += 1
    
    # Analyze components
    components = list(nx.connected_components(G_sub))
    largest = max(components, key=len) if components else set()
    
    # Calculate average component size (excluding largest)
    other_sizes = [len(c) for c in components if c != largest]
    avg_other_size = np.mean(other_sizes) if other_sizes else 0
    
    return {
        'largest_size': len(largest),
        'num_components': len(components),
        'added_edges': added_edges,
        'edge_density': added_edges / total_edges,
        'avg_other_size': avg_other_size,
        'components': sorted(components, key=len, reverse=True)[:5]  # top 5 largest
    }

logger.info("Setting up simulation parameters...")
# More granular thresholds around expected phase transition
thresholds = np.concatenate([
    np.linspace(0, 0.2, 5),
    np.linspace(0.2, 0.4, 11),  # More points in potential phase transition region
    np.linspace(0.4, 1, 7)
])
results_predicted = []
results_actual = []

# Run simulations for both probability types
for prob_type, results_list in [('predicted', results_predicted), ('actual', results_actual)]:
    logger.info(f"\nRunning simulations for {prob_type} probabilities...")
    
    for i, p in enumerate(thresholds):
        logger.info(f"Threshold {p:.2f} ({i+1}/{len(thresholds)})")
        
        # Increased number of simulations for better averaging
        sim_results = [simulate_fire(G, p, prob_type) for _ in range(50)]
        
        # Calculate averages
        avg_results = {
            'threshold': p,
            'avg_largest': np.mean([r['largest_size'] for r in sim_results]),
            'avg_components': np.mean([r['num_components'] for r in sim_results]),
            'avg_edge_density': np.mean([r['edge_density'] for r in sim_results]),
            'avg_other_size': np.mean([r['avg_other_size'] for r in sim_results])
        }
        
        results_list.append(avg_results)
        logger.info(f"Average largest cluster: {avg_results['avg_largest']:.2f}")

# Create comparison plots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

# Plot 1: Largest cluster sizes
ax1.plot(thresholds, [r['avg_largest'] for r in results_predicted], 'b-', 
         label='Predicted Probabilities')
ax1.plot(thresholds, [r['avg_largest'] for r in results_actual], 'r--',
         label='Actual Fire Probabilities')
ax1.set_xlabel("Probability Threshold")
ax1.set_ylabel("Largest Cluster Size")
ax1.set_title("Largest Cluster Size vs. Probability Threshold")
ax1.legend()
ax1.grid(True)

# Plot 2: Average size of other components
ax2.plot(thresholds, [r['avg_other_size'] for r in results_predicted], 'b-',
         label='Predicted Probabilities')
ax2.plot(thresholds, [r['avg_other_size'] for r in results_actual], 'r--',
         label='Actual Fire Probabilities')
ax2.set_xlabel("Probability Threshold")
ax2.set_ylabel("Average Other Component Size")
ax2.set_title("Average Non-Largest Component Size vs. Probability Threshold")
ax2.legend()
ax2.grid(True)

# Plot 3: Component counts and edge density
ax3.plot(thresholds, [r['avg_components'] for r in results_predicted], 'b-',
         label='Components (Predicted)')
ax3.plot(thresholds, [r['avg_components'] for r in results_actual], 'r--',
         label='Components (Actual)')
ax3.plot(thresholds, [r['avg_edge_density'] for r in results_predicted], 'g-',
         label='Edge Density (Predicted)')
ax3.plot(thresholds, [r['avg_edge_density'] for r in results_actual], 'g--',
         label='Edge Density (Actual)')
ax3.set_xlabel("Probability Threshold")
ax3.set_ylabel("Count / Density")
ax3.set_title("Components and Edge Density vs. Probability Threshold")
ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(root_dir, "data", "percolation_comparison.png"))
logger.info("Analysis complete!")