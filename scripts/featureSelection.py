import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os

# Create results directory for feature selection
os.makedirs('results/feature_selection', exist_ok=True)

def load_data(filepath='results/preprocessing/pokedex_engineered_features.csv'):
    """
    Load the preprocessed Pokémon dataset
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the data for feature selection
    """
    # Select all numeric features
    numeric_features = [
        'total_stats', 'physical_bias', 'offensive_bias', 
        'stat_variance', 'weight_height_ratio', 
        'hp', 'attack', 'defense', 's_attack', 's_defense', 'speed',
        'evo_set_size', 'height', 'weight'  # Added original features too
    ]
    
    # Prepare features
    X = df[numeric_features]
    
    # Prepare target (primary type)
    df['primary_type'] = df['type'].str.split('/').str[0].str.strip()
    
    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(df['primary_type'])
    
    return X, y, numeric_features, le

def select_features(X, y, numeric_features):
    """
    Perform feature selection using Random Forest importance
    """
    # Split the data for training the feature selector
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train a Random Forest model
    print("Training Random Forest for feature selection...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    
    # Get feature importance
    importances = rf.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': numeric_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance for Pokémon Type Prediction')
    plt.tight_layout()
    plt.savefig('results/feature_selection/feature_importance.png', dpi=300)
    
    # Save feature importance to CSV
    feature_importance.to_csv('results/feature_selection/feature_importance.csv', index=False)
    
    # Select top N features (e.g., top 7)
    top_n = 7  # Adjust this number as needed
    selected_features = feature_importance['Feature'].iloc[:top_n].tolist()
    
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Save selected features to a text file
    with open('results/feature_selection/selected_features.txt', 'w') as f:
        f.write("Selected features:\n")
        for feature in selected_features:
            f.write(f"- {feature}\n")
    
    return selected_features, feature_importance

def main():
    """
    Main function to run the feature selection
    """
    print("\n=== FEATURE SELECTION USING RANDOM FOREST IMPORTANCE ===")
    
    # Load the data
    df = load_data()
    
    # Preprocess the data
    X, y, numeric_features, le = preprocess_data(df)
    
    # Select features
    selected_features, feature_importance = select_features(X, y, numeric_features)
    
    print("\n=== FEATURE SELECTION COMPLETE ===")
    print(f"Results saved to results/feature_selection/")

if __name__ == "__main__":
    main()