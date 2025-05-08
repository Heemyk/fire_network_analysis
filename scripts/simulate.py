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
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

logger.info("Starting simulation process...")
logger.info("Loading graph from file...")
G = nx.read_gml(os.path.join(root_dir, "data", "fire_percolation_graph.gml"))
logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

def simulate_fire(G, p_threshold):
    """Simulate one run of fire spread at threshold p."""
    G_sub = nx.Graph()
    G_sub.add_nodes_from(G.nodes(data=True))
    
    added_edges = 0
    total_edges = G.number_of_edges()
    high_risk_edges = 0
    
    for u, v, data in G.edges(data=True):
        edge_prob = data['p']
        if edge_prob >= p_threshold:
            if random.random() < edge_prob:
                G_sub.add_edge(u, v)
                added_edges += 1
                # Count edges between high-risk nodes
                if (G.nodes[u]['actual_fire_prob'] > 2.0 and 
                    G.nodes[v]['actual_fire_prob'] > 2.0):
                    high_risk_edges += 1
    
    # Analyze components
    components = list(nx.connected_components(G_sub))
    
    # Get high-risk components (components containing at least one high-risk node)
    high_risk_components = []
    for comp in components:
        if any(G.nodes[node]['actual_fire_prob'] > 2.0 for node in comp):
            high_risk_components.append(comp)
    
    largest = max(components, key=len)
    largest_high_risk = max(high_risk_components, key=len) if high_risk_components else set()
    
    # Get top 5 largest components
    sorted_components = sorted(components, key=len, reverse=True)[:5]
    sorted_high_risk = sorted(high_risk_components, key=len, reverse=True)[:5]
    
    return {
        'largest_size': len(largest),
        'largest_high_risk_size': len(largest_high_risk),
        'num_components': len(components),
        'num_high_risk_components': len(high_risk_components),
        'added_edges': added_edges,
        'high_risk_edges': high_risk_edges,
        'edge_density': added_edges / total_edges,
        'top_components': sorted_components,
        'top_high_risk_components': sorted_high_risk
    }

def save_components_to_csv(components, threshold, component_type, run_number):
    """Save component grid IDs to CSV with additional metadata."""
    rows = []
    for comp_idx, component in enumerate(components):
        for grid_id in component:
            rows.append({
                'GRID_ID': grid_id,
                'component_size': len(component),
                'component_rank': comp_idx + 1,  # 1-based ranking
                'threshold': threshold,
                'component_type': component_type,
                'run_number': run_number,
                'fire_prob': G.nodes[grid_id]['actual_fire_prob'],
                'predicted_prob': G.nodes[grid_id]['predicted_fire_prob']
            })
    return rows

logger.info("Setting up simulation parameters...")
# Key thresholds to analyze in detail
key_thresholds = [0.4]
# Full range for plots
# plot_thresholds = np.linspace(0, 1, 21)
plot_thresholds = [0.4]
results = []
component_data = []

# Run simulations
logger.info("Starting simulations...")
for i, p in enumerate(plot_thresholds):
    logger.info(f"Running simulations for threshold {p:.2f} ({i+1}/{len(plot_thresholds)})")
    
    # Run multiple simulations
    sim_results = []
    for run in range(10):  # Reduced number of runs for storage efficiency
        result = simulate_fire(G, p)
        sim_results.append(result)
        
        # Store component data for key thresholds
        if p in key_thresholds:
            # Save regular components
            component_data.extend(save_components_to_csv(
                result['top_components'], p, 'all', run))
            # Save high-risk components
            component_data.extend(save_components_to_csv(
                result['top_high_risk_components'], p, 'high_risk', run))
    
    # Calculate averages
    avg_results = {
        'threshold': p,
        'avg_largest': np.mean([r['largest_size'] for r in sim_results]),
        'avg_largest_high_risk': np.mean([r['largest_high_risk_size'] for r in sim_results]),
        'avg_components': np.mean([r['num_components'] for r in sim_results]),
        'avg_high_risk_components': np.mean([r['num_high_risk_components'] for r in sim_results]),
        'avg_edge_density': np.mean([r['edge_density'] for r in sim_results]),
        'avg_high_risk_edges': np.mean([r['high_risk_edges'] for r in sim_results])
    }
    
    results.append(avg_results)
    
    logger.info(f"Average largest cluster size: {avg_results['avg_largest']:.2f}")
    logger.info(f"Average largest high-risk cluster: {avg_results['avg_largest_high_risk']:.2f}")
    logger.info(f"Edge density: {avg_results['avg_edge_density']:.3f}")

# Save component data to CSV
logger.info("Saving component data to CSV...")
df_components = pd.DataFrame(component_data)
output_path = os.path.join(root_dir, "data", "component_analysis0pt4.csv")
df_components.to_csv(output_path, index=False)
logger.info(f"Saved component data to {output_path}")

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot 1: Cluster sizes
ax1.plot(plot_thresholds, [r['avg_largest'] for r in results], 'b-', 
         label='Largest Cluster (All)')
ax1.plot(plot_thresholds, [r['avg_largest_high_risk'] for r in results], 'r--',
         label='Largest High-Risk Cluster')
ax1.set_xlabel("Probability Threshold")
ax1.set_ylabel("Cluster Size")
ax1.set_title("Cluster Size vs. Probability Threshold")
ax1.legend()
ax1.grid(True)

# Plot 2: Component counts and edge density
ax2.plot(plot_thresholds, [r['avg_components'] for r in results], 'b-',
         label='Total Components')
ax2.plot(plot_thresholds, [r['avg_high_risk_components'] for r in results], 'r--',
         label='High-Risk Components')
ax2.plot(plot_thresholds, [r['avg_edge_density'] for r in results], 'g-',
         label='Edge Density')
ax2.set_xlabel("Probability Threshold")
ax2.set_ylabel("Count / Density")
ax2.set_title("Components and Edge Density vs. Probability Threshold")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(root_dir, "data", "percolation_analysis_2.png"))
logger.info("Analysis complete!")
