import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory if it doesn't exist
os.makedirs('results/figures/scatter_plots', exist_ok=True)

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

def generate_scatter_plots(df, target_column='type'):
    """
    Generate scatter plots between key numeric features and a target variable
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
        target_column (str): The target column to analyze against
    """
    print(f"\n=== GENERATING SCATTER PLOTS WITH TARGET: {target_column} ===")
    
    # Get numeric columns (excluding id and evo_set which are more identifiers than features)
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_columns = [col for col in numeric_columns if col not in ['id', 'evo_set']]
    
    # Create scatter plots for each pair of numeric features
    for i, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[i+1:]:  # Start from i+1 to avoid duplicates
            print(f"Generating scatter plot for {col1} vs {col2}...")
            
            plt.figure(figsize=(10, 8))
            
            # If target is categorical (like type), create a colored scatter plot
            if df[target_column].dtype == 'object':
                # Get the top 10 most common types to avoid too many colors
                top_types = df[target_column].value_counts().nlargest(10).index
                
                # Filter data for top types
                plot_data = df[df[target_column].isin(top_types)]
                
                # Create a scatter plot with colors based on type
                sns.scatterplot(
                    data=plot_data,
                    x=col1,
                    y=col2,
                    hue=target_column,
                    alpha=0.7,
                    s=80  # Point size
                )
                plt.legend(title=target_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # If target is numeric, use a color gradient
                scatter = plt.scatter(
                    df[col1],
                    df[col2],
                    c=df[target_column],
                    alpha=0.7,
                    s=80,
                    cmap='viridis'
                )
                plt.colorbar(scatter, label=target_column)
            
            # Add title and labels
            plt.title(f'Relationship between {col1} and {col2} by {target_column}', fontsize=14)
            plt.xlabel(col1, fontsize=12)
            plt.ylabel(col2, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            sns.regplot(
                x=col1,
                y=col2,
                data=df,
                scatter=False,
                line_kws={"color": "red", "alpha": 0.7, "lw": 2}
            )
            
            # Improve layout
            plt.tight_layout()
            
            # Save figure
            output_path = f'results/figures/scatter_plots/{col1}_vs_{col2}_by_{target_column}.png'
            plt.savefig(output_path, dpi=300)
            print(f"Saved to {output_path}")
            
            # Close figure to free memory
            plt.close()
    
    print(f"\nAll scatter plots saved to results/figures/scatter_plots/")

def main():
    """Main function to generate scatter plots"""
    # Load the data
    df = load_data()
    
    # Generate scatter plots with type as target
    generate_scatter_plots(df, target_column='type')
    
    # You can also try with a numeric target
    # generate_scatter_plots(df, target_column='hp')
    
    print("\n=== SCATTER PLOT GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()