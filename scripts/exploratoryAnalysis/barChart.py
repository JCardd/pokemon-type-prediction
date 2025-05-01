import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory if it doesn't exist
os.makedirs('results/figures/bar_charts', exist_ok=True)

def load_data(filepath='data/pokedex.csv'):
    """
    Load the Pokédex dataset from CSV file
    
    Args:
        filepath (str): Path to the dataset file
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def generate_total_stats_bar_chart(df):
    """
    Generate bar chart for total base stats ranges
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== GENERATING BAR CHART FOR TOTAL BASE STATS ===")
    
    # Calculate total base stats
    df['total_stats'] = df['hp'] + df['attack'] + df['defense'] + df['s_attack'] + df['s_defense'] + df['speed']
    
    # Create stat ranges for total base stats
    df['total_stats_range'] = pd.cut(df['total_stats'], 
                                    bins=[0, 300, 400, 500, 600, 800], 
                                    labels=['Very Low (0-300)', 'Low (301-400)', 'Medium (401-500)', 
                                           'High (501-600)', 'Very High (601+)'])
    
    total_stats_counts = df['total_stats_range'].value_counts().sort_index()
    
    # Convert to DataFrame for seaborn
    plot_df = pd.DataFrame({
        'Range': total_stats_counts.index,
        'Count': total_stats_counts.values
    })
    
    plt.figure(figsize=(12, 7))
    
    # Fixed version using hue parameter properly
    bars = sns.barplot(x='Range', y='Count', hue='Range', data=plot_df, palette='viridis', legend=False)
    
    # Add count labels on top of bars
    for i, count in enumerate(total_stats_counts.values):
        bars.text(i, count + 5, str(count), ha='center')
    
    plt.title('Distribution of Pokémon by Total Base Stats', fontsize=16)
    plt.xlabel('Total Base Stats Range', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    output_path = 'results/figures/bar_charts/total_stats_distribution.png'
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")
    plt.close()
    
    print(f"\nTotal base stats bar chart saved to results/figures/bar_charts/")

def main():
    """Main function to generate bar chart for total base stats"""
    # Load the data
    df = load_data()
    
    # Generate total base stats bar chart
    generate_total_stats_bar_chart(df)
    
    print("\n=== BAR CHART GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()