import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import joblib
import os
import logging
from scipy.stats import sem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

def load_base_network():
    """Load the base network structure."""
    logger.info("Loading base network...")
    df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
    neighbors_df = pd.read_csv(os.path.join(root_dir, "data", "hex_neighbors.csv"))
    
    # Create base graph structure
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['GRID_ID'], veg=row['MEAN_VEG'])
    
    for _, row in neighbors_df.iterrows():
        G.add_edge(row['GRID_ID'], row['GRID_ID_1'])
    
    return G, df['MEAN_VEG'].values

def simulate_percolation(G, fire_probs, num_trials=50):
    """Run percolation simulation with given fire probabilities."""
    # Assign edge probabilities as average of node probabilities
    for u, v in G.edges():
        prob_u = fire_probs[list(G.nodes()).index(u)]
        prob_v = fire_probs[list(G.nodes()).index(v)]
        G.edges[u, v]['p'] = (prob_u + prob_v) / 2
    
    # Run percolation for different thresholds
    thresholds = np.linspace(0, 1, 21)
    results = []
    
    for p in thresholds:
        logger.info(f"Running percolation with threshold {p:.2f}")
        trial_sizes = []
        
        for _ in range(num_trials):
            # Create subgraph based on threshold
            G_sub = nx.Graph()
            G_sub.add_nodes_from(G.nodes(data=True))
            
            for u, v in G.edges():
                if random.random() < p:
                    G_sub.add_edge(u, v)
            
            # Get largest component size
            components = list(nx.connected_components(G_sub))
            largest = max(components, key=len)
            trial_sizes.append(len(largest))
        
        results.append({
            'threshold': p,
            'mean_size': np.mean(trial_sizes),
            'std_size': np.std(trial_sizes),
            'sem_size': sem(trial_sizes)
        })
    
    return results

def run_scenario(scenario_name, veg_change, base_veg):
    """Run a complete scenario with vegetation change."""
    logger.info(f"Running scenario: {scenario_name}")
    
    # Load model and scaler
    model = joblib.load(os.path.join(root_dir, "data", "fire_model.joblib"))
    scaler = joblib.load(os.path.join(root_dir, "data", "fire_scaler.joblib"))
    
    # Apply vegetation change
    new_veg = base_veg * (1 + veg_change)
    new_veg_scaled = scaler.transform(new_veg.reshape(-1, 1))
    
    # Predict fire probabilities
    fire_probs = model.predict(new_veg_scaled)
    
    # Run percolation
    G = load_base_network()[0]
    results = simulate_percolation(G, fire_probs)
    
    return results

def plot_scenario_comparison(scenarios):
    """Plot comparison of different scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for name, results in scenarios.items():
        thresholds = [r['threshold'] for r in results]
        means = [r['mean_size'] for r in results]
        sems = [r['sem_size'] for r in results]
        
        ax.plot(thresholds, means, '-', label=name)
        ax.fill_between(thresholds, 
                       [m - s for m, s in zip(means, sems)],
                       [m + s for m, s in zip(means, sems)],
                       alpha=0.2)
    
    ax.set_xlabel("Bond Probability")
    ax.set_ylabel("Largest Component Size")
    ax.set_title("Percolation Behavior Under Different Vegetation Scenarios")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "data", "scenario_comparison.png"))
    logger.info("Saved scenario comparison plot")

def main():
    # Load base network and vegetation data
    G, base_veg = load_base_network()
    
    # Define scenarios
    scenarios = {
        'Baseline': 0.0,
        'Vegetation -25%': -0.25,
        'Vegetation +25%': 0.25,
        'Vegetation -50%': -0.50,
        'Vegetation +50%': 0.50
    }
    
    # Run all scenarios
    results = {}
    for name, veg_change in scenarios.items():
        results[name] = run_scenario(name, veg_change, base_veg)
    
    # Plot comparison
    plot_scenario_comparison(results)
    
    # Save results
    np.save(os.path.join(root_dir, "data", "percolation_scenarios.npy"), results)
    logger.info("Saved scenario results")

if __name__ == "__main__":
    main() 