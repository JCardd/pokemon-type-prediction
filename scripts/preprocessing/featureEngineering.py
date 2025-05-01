import pandas as pd
import numpy as np
import os

# Create results directory if it doesn't exist
os.makedirs('results/preprocessing', exist_ok=True)

def load_data(filepath='data/pokedex.csv'):
    """
    Load the Pokédex dataset
    
    Args:
        filepath (str): Path to the dataset
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def engineer_features(df):
    """
    Engineer new features from the dataset
    
    Args:
        df (pandas.DataFrame): The original dataset
        
    Returns:
        pandas.DataFrame: Dataset with new features
    """
    print("\n=== ENGINEERING NEW FEATURES ===")
    
    # Make a copy of the dataframe
    enhanced_df = df.copy()
    
    # 1. Create total stats feature
    print("Creating total_stats feature...")
    enhanced_df['total_stats'] = enhanced_df['hp'] + enhanced_df['attack'] + enhanced_df['defense'] + \
                                enhanced_df['s_attack'] + enhanced_df['s_defense'] + enhanced_df['speed']
    
    # 2. Create physical vs special bias feature
    print("Creating physical_bias feature...")
    enhanced_df['physical_bias'] = (enhanced_df['attack'] + enhanced_df['defense']) / \
                                   (enhanced_df['s_attack'] + enhanced_df['s_defense'])
    
    # 3. Create offensive vs defensive bias feature
    print("Creating offensive_bias feature...")
    enhanced_df['offensive_bias'] = (enhanced_df['attack'] + enhanced_df['s_attack']) / \
                                     (enhanced_df['defense'] + enhanced_df['s_defense'])
    
    # 4. Create stat variance feature (how specialized vs balanced a Pokemon is)
    print("Creating stat_variance feature...")
    stat_columns = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
    enhanced_df['stat_variance'] = enhanced_df[stat_columns].apply(np.var, axis=1)
    
    # 5. Create BMI-like feature (weight to height ratio)
    print("Creating weight_height_ratio feature...")
    enhanced_df['weight_height_ratio'] = enhanced_df['weight'] / enhanced_df['height']
    
    # 6. Create evolution stage indicators
    print("Creating evolution stage features...")
    # Group by evolution set and count members in each set
    evo_set_sizes = enhanced_df.groupby('evo_set')['name'].count().reset_index()
    evo_set_sizes.columns = ['evo_set', 'evo_set_size']
    
    # Merge this information back to the main dataframe
    enhanced_df = enhanced_df.merge(evo_set_sizes, on='evo_set', how='left')
    
    # 7. Create stat categories (e.g., low, medium, high)
    print("Creating stat category features...")
    for stat in stat_columns:
        enhanced_df[f'{stat}_category'] = pd.qcut(
            enhanced_df[stat], 
            q=[0, 0.25, 0.5, 0.75, 1.0], 
            labels=['Very Low', 'Low', 'Medium', 'High']
        )
    
    enhanced_df['total_stats_category'] = pd.qcut(
        enhanced_df['total_stats'], 
        q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )
    
    # Save the enhanced dataset
    enhanced_df.to_csv('results/preprocessing/pokedex_engineered_features.csv', index=False)
    print("Enhanced dataset saved to results/preprocessing/pokedex_engineered_features.csv")
    
    return enhanced_df

def main():
    """Main function to engineer features"""
    # Load the data
    df = load_data()
    
    # Quick check for missing values
    missing_values = df.isnull().sum().sum()
    print(f"Total missing values in dataset: {missing_values}")
    
    # Engineer new features
    enhanced_df = engineer_features(df)
    
    # Display the new features
    print("\nNew features added:")
    original_columns = set(df.columns)
    new_columns = set(enhanced_df.columns) - original_columns
    for col in sorted(new_columns):
        print(f"- {col}")
    
    print("\n=== FEATURE ENGINEERING COMPLETE ===")

if __name__ == "__main__":
    main()