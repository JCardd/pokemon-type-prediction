import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    f1_score, 
    precision_score, 
    recall_score
)
import os

# Create results directory for modeling if it doesn't exist
os.makedirs('results/modeling/naive_bayes_selected', exist_ok=True)

def load_data(filepath='results/preprocessing/pokedex_engineered_features.csv'):
    """
    Load the preprocessed Pokémon dataset
    
    Args:
        filepath (str): Path to the preprocessed dataset
        
    Returns:
        pandas.DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    return pd.read_csv(filepath)

def preprocess_data(df):
    """
    Preprocess the data for Naive Bayes classification with selected features
    
    Args:
        df (pandas.DataFrame): Input dataframe
        
    Returns:
        tuple: X (features), y (target), label encoder
    """
    # Select only the features identified by feature selection
    selected_features = [
        'physical_bias', 
        'weight_height_ratio', 
        'offensive_bias', 
        'stat_variance', 
        'weight', 
        'speed', 
        's_attack'
    ]
    
    # Prepare features using only selected features
    X = df[selected_features]
    
    # Prepare target (primary type)
    df['primary_type'] = df['type'].str.split('/').str[0].str.strip()
    
    # Identify and filter out classes with only one sample
    type_counts = df['primary_type'].value_counts()
    types_to_keep = type_counts[type_counts >= 2].index
    df = df[df['primary_type'].isin(types_to_keep)]
    
    # Re-select features after filtering
    X = df[selected_features]
    
    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(df['primary_type'])
    
    return X, y, le

def train_naive_bayes(X, y):
    """
    Train Naive Bayes model with cross-validation
    
    Args:
        X (pandas.DataFrame): Features
        y (numpy.array): Target variable
        
    Returns:
        tuple: Trained model, test data, predictions, cross-validation scores, scaler
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train Gaussian Naive Bayes model
    clf = GaussianNB()
    
    # Fit the model
    clf.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = clf.predict(X_test_scaled)
    
    # Perform cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=2)
    
    return clf, X_test_scaled, y_test, y_pred, cv_scores, scaler

def evaluate_model(y_test, y_pred, le):
    """
    Evaluate the Naive Bayes model performance
    
    Args:
        y_test (numpy.array): True labels
        y_pred (numpy.array): Predicted labels
        le (LabelEncoder): Label encoder to map back to original labels
    """
    # Determine the unique labels present in the test and prediction sets
    present_labels = np.union1d(y_test, y_pred)
    target_names = le.inverse_transform(present_labels)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', labels=present_labels, zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', labels=present_labels, zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', labels=present_labels, zero_division=0)
    
    # Generate classification report text
    report_text = classification_report(y_test, y_pred, 
                                        labels=present_labels, 
                                        target_names=target_names, 
                                        zero_division=0)
    
    # Print classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(report_text)
    
    # Save classification report
    with open('results/modeling/naive_bayes_selected/classification_report.txt', 'w') as f:
        f.write("=== CLASSIFICATION REPORT ===\n")
        f.write(report_text)
    
    # Save model performance metrics
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }
    
    with open('results/modeling/naive_bayes_selected/performance_metrics.txt', 'w') as f:
        f.write("=== MODEL PERFORMANCE METRICS ===\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
    
    return metrics

def main():
    """
    Main function to run the Naive Bayes classification model with selected features
    """
    print("\n=== NAIVE BAYES CLASSIFICATION MODEL WITH SELECTED FEATURES ===")
    
    # Load the data
    df = load_data()
    
    # Preprocess the data
    X, y, le = preprocess_data(df)
    
    # Train the model
    clf, X_test_scaled, y_test, y_pred, cv_scores, scaler = train_naive_bayes(X, y)
    
    # Evaluate the model
    metrics = evaluate_model(y_test, y_pred, le)
    
    # Print cross-validation results
    print("\n=== CROSS-VALIDATION RESULTS ===")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Save cross-validation results
    with open('results/modeling/naive_bayes_selected/cross_validation_results.txt', 'w') as f:
        f.write("=== CROSS-VALIDATION RESULTS ===\n")
        f.write(f"Cross-validation scores: {cv_scores}\n")
        f.write(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n=== NAIVE BAYES MODEL WITH SELECTED FEATURES ANALYSIS COMPLETE ===")

if __name__ == "__main__":
    main()