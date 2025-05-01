import pandas as pd
import os

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)

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

def analyze_basic_info(df):
    """
    Analyze and print basic information about the dataset
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
        
    Returns:
        dict: Dictionary containing basic dataset information
    """
    # Get number of instances and attributes
    num_instances = df.shape[0]
    num_attributes = df.shape[1]
    
    print("\n=== DATASET BASIC INFORMATION ===")
    print(f"Number of instances (rows): {num_instances}")
    print(f"Number of attributes (columns): {num_attributes}")
    
    # Print DataFrame info
    print("\n=== DATAFRAME INFO ===")
    df.info()
    
    # Print DataFrame description (numeric columns)
    print("\n=== DATAFRAME DESCRIPTION (NUMERIC COLUMNS) ===")
    print(df.describe())
    
    # Save describe output to CSV
    df.describe().to_csv('results/numeric_stats.csv')
    print("Numeric statistics saved to results/numeric_stats.csv")
    
    # Get data types information
    data_types = df.dtypes.value_counts()
    print("\n=== DATA TYPES ===")
    for dtype, count in data_types.items():
        print(f"{dtype}: {count} columns")
    
    # Create a summary dictionary
    summary = {
        'num_instances': num_instances,
        'num_attributes': num_attributes,
        'data_types': data_types.to_dict(),
        'source': 'Kaggle - Pokédex for All 1025 Pokémon',
        'source_url': 'https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description'
    }
    
    return summary

def analyze_columns(df):
    """
    Analyze and print information about each column in the dataset
    
    Args:
        df (pandas.DataFrame): The dataset to analyze
    """
    print("\n=== COLUMN INFORMATION ===")
    
    # Get basic statistics for each column
    for col in df.columns:
        print(f"\nColumn: {col}")
        print(f"  - Type: {df[col].dtype}")
        print(f"  - Missing values: {df[col].isna().sum()} ({df[col].isna().mean():.2%})")
        
        if df[col].dtype == 'object':
            # For categorical/text columns
            print(f"  - Unique values: {df[col].nunique()}")
            if df[col].nunique() < 10:  # Only show values if there are few unique ones
                print(f"  - Values: {df[col].unique()}")
            elif col == 'type':  # Special case for Pokémon types
                print(f"  - Most common types: {df[col].value_counts().head(5).to_dict()}")
        else:
            # For numeric columns
            print(f"  - Min: {df[col].min()}")
            print(f"  - Max: {df[col].max()}")
            print(f"  - Mean: {df[col].mean()}")
            print(f"  - Median: {df[col].median()}")

def generate_dataset_description(df, summary):
    """
    Generate a formatted dataset description for the report
    
    Args:
        df (pandas.DataFrame): The dataset
        summary (dict): Dataset summary information
        
    Returns:
        str: Formatted dataset description
    """
    # Create a description string
    description = "## Dataset Description\n\n"
    
    # Basic information
    description += f"The dataset contains information about {summary['num_instances']} Pokémon with {summary['num_attributes']} attributes.\n\n"
    
    # Source information
    description += f"**Source**: {summary['source']}\n"
    description += f"**URL**: {summary['source_url']}\n\n"
    
    # Data types
    description += "### Data Types\n"
    for dtype, count in summary['data_types'].items():
        description += f"- {dtype}: {count} columns\n"
    description += "\n"
    
    # Column descriptions
    description += "### Attributes\n"
    
    # Define column descriptions
    column_descriptions = {
        'id': "Unique identifier for each Pokémon",
        'name': "Name of the Pokémon",
        'height': "Height of the Pokémon in decimeters",
        'weight': "Weight of the Pokémon in hectograms",
        'hp': "Hit Points - a measure of health",
        'attack': "Base Attack stat",
        'defense': "Base Defense stat",
        's_attack': "Base Special Attack stat",
        's_defense': "Base Special Defense stat",
        'speed': "Base Speed stat",
        'type': "Pokémon type(s)",
        'evo_set': "Evolution set identifier",
        'info': "Text description of the Pokémon"
    }
    
    # Add column information with descriptions
    for col in df.columns:
        description += f"- **{col}**: {column_descriptions.get(col, 'No description available')}\n"
        if df[col].dtype != 'object':
            description += f"  - Numeric attribute (Range: {df[col].min()} to {df[col].max()})\n"
        else:
            description += f"  - Categorical/Text attribute (Unique values: {df[col].nunique()})\n"
    
    # Save to file
    with open('results/dataset_description.md', 'w') as f:
        f.write(description)
    
    print(f"\nDataset description saved to results/dataset_description.md")
    return description

def save_summary(summary, filepath='results/dataset_summary.txt'):
    """
    Save the dataset summary to a text file
    
    Args:
        summary (dict): Dictionary containing dataset summary
        filepath (str): Path to save the summary
    """
    with open(filepath, 'w') as f:
        f.write("=== POKÉDEX DATASET SUMMARY ===\n\n")
        f.write(f"Number of instances (rows): {summary['num_instances']}\n")
        f.write(f"Number of attributes (columns): {summary['num_attributes']}\n\n")
        
        f.write("=== DATA TYPES ===\n")
        for dtype, count in summary['data_types'].items():
            f.write(f"{dtype}: {count} columns\n")
        
        f.write("\n=== SOURCE INFORMATION ===\n")
        f.write(f"Source: {summary['source']}\n")
        f.write(f"URL: {summary['source_url']}\n")

def main():
    """Main function to run the exploratory analysis"""
    # Load the data
    df = load_data()
    
    # Display the first few rows of the dataset
    print("\n=== FIRST 5 ROWS OF THE DATASET ===")
    print(df.head())
    
    # Analyze basic information
    summary = analyze_basic_info(df)
    
    # Analyze columns
    analyze_columns(df)
    
    # Generate and save dataset description for the report
    dataset_description = generate_dataset_description(df, summary)
    print("\n=== DATASET DESCRIPTION ===")
    print(dataset_description)
    
    # Save summary to file
    save_summary(summary)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Summary saved to results/dataset_summary.txt")

if __name__ == "__main__":
    main()