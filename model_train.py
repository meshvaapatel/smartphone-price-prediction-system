"""
Smartphone Price Prediction Model Training Script
-------------------------------------------------
This script trains a Random Forest Regression model to predict smartphone prices
based on key specifications and features from Flipkart data.

Steps:
1. Load and prepare the cleaned dataset
2. Select final training features
3. Create preprocessing pipeline (scaling + encoding)
4. Define Random Forest model with hyperparameter tuning
5. Perform RandomizedSearchCV for optimization
6. Evaluate and save the best model

"""

# --- Imports ---
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

# 1. Load the Cleaned Dataset
try:
    df = pd.read_csv('flipkart_smartphones_cleaned.csv')
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: 'flipkart_smartphones_cleaned.csv' not found.")
    print("Please ensure the dataset is in the same directory as this script or provide the full path.")
    exit()

# 2. Select Final Features for Model Training
# These features were selected based on domain understanding and exploratory data analysis.
final_features = [
    'Brand', 'RAM_GB', 'ROM_GB', 'Display_Size_inch', 'Display_Type',
    'Battery_mAh', 'Processor', 'Warranty_Years'
]

print(f"\nUsing final features for training: {final_features}")

# Keep only selected features + target variable
df_final = df[final_features + ['Price']].copy()
df_final.dropna(inplace=True)  # Ensure no missing values

# Define feature matrix (X) and target vector (y)
X = df_final[final_features]
y = df_final['Price']

# 3. Build a Preprocessing Pipeline
# Separate categorical and numerical features for appropriate transformations
categorical_features = X.select_dtypes(include=['object', 'category']).columns
numerical_features = X.select_dtypes(include=['number']).columns

print(f"\nIdentified Categorical Features: {categorical_features.tolist()}")
print(f"Identified Numerical Features: {numerical_features.tolist()}")

# Create column transformer for preprocessing
# - Numerical features → StandardScaler
# - Categorical features → OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)
print("\n⚙️ Preprocessing pipeline created successfully!")

# 4. Define Model and Hyperparameter Search Space
# Random Forest chosen for its robustness and performance on tabular data
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# Combine preprocessing and model into a single pipeline
ml_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', rf_model)
])

# Define hyperparameter space for RandomizedSearchCV
# Keys must be prefixed with the pipeline step name ('regressor__')
param_dist = {
    'regressor__n_estimators': [100, 200, 300, 500, 700],
    'regressor__max_depth': [10, 15, 20, 25, 30, None],
    'regressor__min_samples_split': [2, 5, 10, 15],
    'regressor__min_samples_leaf': [1, 2, 4, 6],
    'regressor__max_features': ['sqrt', 'log2', 1.0]  # 1.0 means all features
}

# 5. Train-Test Split and Hyperparameter Tuning
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nStarting hyperparameter tuning using RandomizedSearchCV...")

# RandomizedSearchCV parameters:
# - n_iter: number of random parameter combinations to try
# - cv: number of cross-validation folds
# - scoring: metric to optimize (R² for regression)
# - n_jobs=-1: use all available CPU cores
random_search = RandomizedSearchCV(
    estimator=ml_pipeline,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    scoring='r2'
)

# Run the search
random_search.fit(X_train, y_train)
print("\nHyperparameter tuning complete!")

# 6. Evaluate the Best Model
print("\n🏆 Best Parameters Found:")
print(random_search.best_params_)

best_model = random_search.best_estimator_
r2_score = best_model.score(X_test, y_test)
print(f"\nBest Model R² Score on Test Data: {r2_score:.4f}")

# 7. Save the Final Trained Model
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)   

print("\nFinal best model pipeline saved successfully as 'random_forest_model.pkl'!")

print("\n🎯 Model training pipeline completed successfully!")
