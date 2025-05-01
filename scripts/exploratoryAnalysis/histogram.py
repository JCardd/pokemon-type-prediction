import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory if it doesn't exist
os.makedirs('results/figures/histograms', exist_ok=True)

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

def generate_histograms(df):
    """
    Generate histograms for numeric columns and save to results/figures/histograms
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== GENERATING HISTOGRAMS FOR NUMERIC COLUMNS ===")
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Set figure style
    sns.set_style("whitegrid")
    
    # Generate histogram for each numeric column
    for col in numeric_columns:
        print(f"Generating histogram for {col}...")
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram with KDE
        sns.histplot(df[col], kde=True)
        
        # Add title and labels
        plt.title(f'Distribution of {col}', fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Improve layout
        plt.tight_layout()
        
        # Save figure
        output_path = f'results/figures/histograms/{col}_histogram.png'
        plt.savefig(output_path, dpi=300)
        print(f"Saved to {output_path}")
        
        # Close figure to free memory
        plt.close()
    
    print(f"\nAll histograms saved to results/figures/histograms/")

def main():
    """Main function to generate histograms"""
    # Load the data
    df = load_data()
    
    # Generate histograms
    generate_histograms(df)
    
    print("\n=== HISTOGRAM GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()