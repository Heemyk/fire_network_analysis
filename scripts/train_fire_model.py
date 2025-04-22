import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (root of the project)
root_dir = os.path.dirname(script_dir)

def load_and_prepare_data():
    """Load and prepare data for modeling."""
    logger.info("Loading fire data...")
    df = pd.read_csv(os.path.join(root_dir, "data", "fire_data.csv"))
    
    # Prepare features and target
    X = df[['MEAN_VEG']].values
    # Convert fire probabilities to binary (1 for high risk, 0 for low risk)
    FIRE_THRESHOLD = 200  # Based on the clear separation in the data
    y = (df['MEAN_FIRE'] > FIRE_THRESHOLD).astype(int)
    
    logger.info(f"High risk cells: {y.sum()} ({y.mean():.1%})")
    
    return X, y, df['GRID_ID'].values

def train_model(X, y):
    """Train the model and evaluate its performance."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    logger.info("Training Random Forest classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Get predictions and probabilities
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return model, scaler, (X_test_scaled, y_test, y_prob)

def plot_model_performance(model, scaler, X, y, test_data):
    """Create plots showing model performance."""
    X_test, y_test, y_prob = test_data
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic')
    ax1.legend()
    ax1.grid(True)
    
    # Probability distribution by vegetation
    X_grid = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
    X_grid_scaled = scaler.transform(X_grid)
    y_grid_prob = model.predict_proba(X_grid_scaled)[:, 1]
    
    # Sort data for better visualization
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    ax2.scatter(X, y, alpha=0.1, color='lightblue', label='Actual')
    ax2.plot(X_grid, y_grid_prob, 'r-', label='Predicted Probability', linewidth=2)
    ax2.set_xlabel("Mean Vegetation")
    ax2.set_ylabel("Fire Risk Probability")
    ax2.set_title("Fire Risk vs. Vegetation")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(root_dir, "data", "fire_model_analysis.png"))
    logger.info("Saved model analysis plots")

def main():
    # Load and prepare data
    X, y, grid_ids = load_and_prepare_data()
    
    # Train model
    model, scaler, test_data = train_model(X, y)
    
    # Plot model performance
    plot_model_performance(model, scaler, X, y, test_data)
    
    # Save model and scaler
    logger.info("Saving model and scaler...")
    joblib.dump(model, os.path.join(root_dir, "data", "fire_model.joblib"))
    joblib.dump(scaler, os.path.join(root_dir, "data", "fire_scaler.joblib"))
    
    # Generate predictions for all data
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'GRID_ID': grid_ids,
        'predicted_fire_prob': probabilities
    })
    predictions_df.to_csv(os.path.join(root_dir, "data", "predicted_fire_probs.csv"), index=False)
    logger.info("Saved predictions")

if __name__ == "__main__":
    main() 