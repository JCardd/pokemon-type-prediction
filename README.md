# Pokémon Type Prediction & Classifier Comparison 📊🐍

![Pokémon Banner](https://via.placeholder.com/1200x300.png?text=Pok%C3%A9mon+ML+Analysis)
*(Optional: Replace the placeholder above with a relevant Pokémon or data analysis image)*

## 📖 Overview

This project analyzes the comprehensive Pokédex dataset (containing information on 1,025 Pokémon up to Generation 9) to predict a Pokémon's primary type based on its base statistics and other attributes. The analysis involves Exploratory Data Analysis (EDA), Feature Engineering, the implementation and comparison of five different machine learning classification algorithms, and an evaluation of the impact of feature selection on model performance.

## 🎯 Project Goal

The primary goal was to explore the Pokédex dataset and determine how effectively various machine learning models could predict a Pokémon's primary type using its statistical attributes. A key component was comparing the performance of different algorithm categories (e.g., linear, tree-based, distance-based, probabilistic, SVM) and assessing whether feature selection improved prediction accuracy for this specific multi-class classification problem.

## 💾 Dataset

*   **Source:** Kaggle - [Pokédex for All 1025 Pokémon w/ Text Description](https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description) by Robert Giza (rzgiza).
*   **Size:** 1,025 instances (Pokémon)
*   **Attributes (Initial):** 13 columns including ID, name, height, weight, base stats (HP, Attack, Defense, S.Attack, S.Defense, Speed), type(s), evolution set ID, and text info.
*   **Target Variable:** Primary Pokémon Type (extracted from the `type` column).
*   **Data Quality:** The dataset had no missing values in the core attributes used.

## ⚙️ Project Workflow

The project followed a standard data science workflow:

1.  **Exploratory Data Analysis (EDA):**
    *   Loaded the raw dataset (`data/pokedex.csv`).
    *   Analyzed basic dataset information (shape, data types, missing values).
    *   Calculated descriptive statistics for numeric features (`results/numeric_stats.csv`).
    *   Generated visualizations (histograms, boxplots, scatter plots) to understand feature distributions and relationships between stats and Pokémon types (`results/figures/`). See `scripts/exploratoryAnalysis/` for the code.

2.  **Feature Engineering:**
    *   Created several new features based on domain knowledge and initial analysis to potentially improve model performance. These include:
        *   `total_stats`: Sum of all six base stats.
        *   `physical_bias`: Ratio of physical stats (Atk+Def) to special stats (S.Atk+S.Def).
        *   `offensive_bias`: Ratio of offensive stats (Atk+S.Atk) to defensive stats (Def+S.Def).
        *   `stat_variance`: Variance across the six base stats (measure of specialization).
        *   `weight_height_ratio`: Simple ratio similar to BMI.
        *   `evo_set_size`: Number of Pokémon in the same evolution family.
        *   *(Stat Categories were also generated but potentially less used in final models)*
    *   The dataset with engineered features was saved (`results/preprocessing/pokedex_engineered_features.csv`). See `scripts/preprocessing/featureEngineering.py`.

3.  **Data Preprocessing (for Modeling):**
    *   Extracted the **primary type** as the target variable (`y`).
    *   Handled the multi-class nature of the target by using `LabelEncoder`.
    *   Filtered out the few Pokémon types with only a single instance to enable stratified splitting and meaningful evaluation.
    *   Selected appropriate numeric features (engineered + original base stats) for input (`X`).
    *   Split data into training and testing sets using `train_test_split` with stratification.
    *   Applied `StandardScaler` to scale features before feeding them into the models.

4.  **Feature Selection:**
    *   Used **Random Forest Feature Importance** on the full set of numeric features (original + engineered) to identify the most predictive attributes for primary type classification.
    *   The top 7 features identified were: `physical_bias`, `weight_height_ratio`, `offensive_bias`, `stat_variance`, `weight`, `speed`, `s_attack`.
    *   Results saved in `results/feature_selection/`. See `scripts/featureSelection.py`.

5.  **Modeling & Comparison:**
    *   Implemented and trained five different classification algorithms from distinct categories:
        1.  Logistic Regression (Linear Model)
        2.  Random Forest (Tree-based Ensemble)
        3.  Support Vector Machine (SVM) (with RBF kernel, balanced class weight)
        4.  K-Nearest Neighbors (KNN) (Distance-based, k=3, distance weighting)
        5.  Gaussian Naive Bayes (Probabilistic)
    *   Each model was trained and evaluated **twice**:
        *   Once using the **full set** of engineered + original numeric features.
        *   Once using only the **top 7 selected features**.
    *   Code for each model (full and selected features) is located in `scripts/modeling/`.

6.  **Evaluation:**
    *   Models were evaluated on the held-out test set using:
        *   Accuracy
        *   Precision (Weighted)
        *   Recall (Weighted)
        *   F1-Score (Weighted)
        *   Detailed Classification Reports (per-class metrics)
    *   Cross-validation (`cv=2`) was performed on the training set to assess model robustness and estimate generalization performance (Mean CV Score +/- Std Dev).
    *   All evaluation metrics and reports were saved for each model variant in the respective subdirectories under `results/modeling/`.

## 📈 Key Findings

*   **Challenge:** Predicting Pokémon primary type based solely on stats is a challenging multi-class classification problem, compounded by significant class imbalance (many types have very few representatives). This resulted in relatively low overall accuracy scores (< 15%) across all models.
*   **Feature Engineering:** Engineered features like `physical_bias`, `weight_height_ratio`, and `offensive_bias` consistently ranked high in feature importance, suggesting they captured meaningful patterns relevant to Pokémon typing.
*   **Feature Selection Impact:**
    *   **Improved Performance:** Random Forest and Naive Bayes showed slight improvements in performance (Accuracy, F1-Score) when using the selected feature set compared to the full set. This aligns with the expectation that these models can benefit from reduced noise and redundancy.
    *   **Decreased Performance:** Logistic Regression, SVM, and KNN performed significantly worse with the reduced feature set. This suggests these models benefited from the information contained in the broader set of features, even those deemed less important by the Random Forest selector, or that the selected features didn't optimally suit their underlying mechanisms (e.g., distance calculations for KNN).
*   **Best Models:** Based on overall performance (considering test metrics and CV scores):
    *   **Random Forest (Selected Features):** Achieved the highest test F1-Score (approx. 11.8%).
    *   **Logistic Regression (All Features):** Had the highest Mean CV Score (approx. 15.4%), indicating potentially better generalization despite slightly lower test set performance in this specific split.
*   **Conclusion:** Feature selection is not universally beneficial and its effectiveness depends heavily on the algorithm used. For this task, ensemble methods (Random Forest) seemed most promising, especially after feature selection, while linear models performed better with access to all available features.

## 🛠️ Tech Stack

*   **Language:** Python 3.x
*   **Core Libraries:**
    *   Pandas (Data manipulation and analysis)
    *   NumPy (Numerical operations)
    *   Scikit-learn (Machine Learning - modeling, preprocessing, evaluation)
    *   Matplotlib (Plotting)
    *   Seaborn (Enhanced plotting)

