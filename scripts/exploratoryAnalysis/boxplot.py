import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(file_path):
    """
    Load the Pokédex dataset from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file
    
    Returns:
    pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_boxplots(df, save_dir="results/figures/boxplots"):
    """
    Create and save boxplots for all numeric attributes in the dataset
    
    Parameters:
    df (pandas.DataFrame): Dataset containing Pokémon attributes
    save_dir (str): Directory to save generated figures
    
    Returns:
    None
    """
    print("Creating boxplots for numeric attributes")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get numeric columns (exclude id and evo_set as they're identifiers)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ['id', 'evo_set']]
    
    # Create individual boxplots for each attribute
    for col in numeric_cols:
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[col])
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(save_dir, f'boxplot_{col}.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved boxplot for {col} to {filename}")
    
    # Create grouped boxplot of all base stats
    stat_cols = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
    
    plt.figure(figsize=(12, 6))
    
    # Melt the dataframe to get stats in long format
    melted_df = pd.melt(df, 
                        id_vars=['name'], 
                        value_vars=stat_cols,
                        var_name='Stat', 
                        value_name='Value')
    
    # Create boxplot with all stats
    sns.boxplot(x='Stat', y='Value', data=melted_df)
    plt.title('Distribution of Base Stats')
    plt.xlabel('Stat')
    plt.ylabel('Value')
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(save_dir, 'boxplot_all_stats.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved combined boxplot for all stats to {filename}")
    
    # Create boxplot for physical attributes
    physical_cols = ['height', 'weight']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, col in enumerate(physical_cols):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f'Distribution of {col.title()}')
        axes[i].set_xlabel(col.title())
    
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(save_dir, 'boxplot_physical_attributes.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved boxplot for physical attributes to {filename}")
    
    print("Completed creating all boxplots")

def create_boxplots_by_type(df, save_dir="results/figures/boxplots_by_type"):
    """
    Create and save boxplots of stats grouped by primary Pokémon type
    
    Parameters:
    df (pandas.DataFrame): Dataset containing Pokémon attributes
    save_dir (str): Directory to save generated figures
    
    Returns:
    None
    """
    print("Creating boxplots of stats grouped by Pokémon type")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Process the type column to extract only the primary type
    # Assuming types are formatted like "Type1" or "Type1/Type2"
    df['primary_type'] = df['type'].str.split('/').str[0].str.strip()
    
    # Get the most common types (top 10) to avoid overcrowding the plots
    top_types = df['primary_type'].value_counts().head(10).index.tolist()
    
    # Filter dataframe to include only Pokémon with these types
    filtered_df = df[df['primary_type'].isin(top_types)].copy()
    
    # Stats to analyze
    stat_cols = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
    
    # Create individual boxplots for each stat by type
    for stat in stat_cols:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='primary_type', y=stat, data=filtered_df)
        plt.title(f'{stat.replace("_", " ").title()} Distribution by Type')
        plt.xlabel('Primary Type')
        plt.ylabel(stat.replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the figure
        filename = os.path.join(save_dir, f'boxplot_{stat}_by_type.png')
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved boxplot for {stat} by type to {filename}")
    
    # Create a figure showing total stats by type
    filtered_df['total_stats'] = filtered_df[stat_cols].sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='primary_type', y='total_stats', data=filtered_df)
    plt.title('Total Base Stats Distribution by Type')
    plt.xlabel('Primary Type')
    plt.ylabel('Total Base Stats')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the figure
    filename = os.path.join(save_dir, 'boxplot_total_stats_by_type.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved boxplot for total stats by type to {filename}")
    
    print("Completed creating all boxplots by type")

def main():
    """Main function to execute the boxplot visualization"""
    print("Starting boxplot visualization process")
    
    # Load the dataset
    df = load_data("data/pokedex.csv")
    
    if df is not None:
        # Create boxplots
        create_boxplots(df)
        
        # Create boxplots by type
        create_boxplots_by_type(df)
        
        print("Boxplot visualization completed successfully")
    else:
        print("Visualization failed due to data loading error")

if __name__ == "__main__":
    main()