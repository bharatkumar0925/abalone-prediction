import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load train and test data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Lowercase column names for consistency
train.columns, test.columns = train.columns.str.lower(), test.columns.str.lower()

# Map categorical variable 'sex' to numerical values
train['sex'] = train['sex'].map({'F': 0, 'M': 1, 'I': 2})
test['sex'] = test['sex'].map({'F': 0, 'M': 1, 'I': 2})

# Feature engineering: Create a new feature 'height weight'
train['height weight'] = train['height'] ** train['shell weight']
test['height weight'] = test['height'] ** test['shell weight']

# Drop 'id' column from the training dataset
train.drop('id', axis=1, inplace=True)

# Split data into features (X) and target (y)
X = train.drop('rings', axis=1)
y = train['rings']


def objective(trial):
    # Define hyperparameters to optimize
    xgb_params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 16),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.05, 0.3),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 0.1, 100.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 0.1, 100.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'subsample': trial.suggest_loguniform('subsample', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
    }

    # Initialize XGBoost model with given hyperparameters
    model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squaredlogerror', **xgb_params)

    # Split data into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    model.fit(X_train, y_train)

    # Make predictions on validation set
    y_pred = model.predict(X_valid)

    # Compute mean squared log error (MSLE) as the score
    score = mean_squared_log_error(y_valid, y_pred)

    return score


# Create Optuna study and optimize hyperparameters
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# Access best hyperparameters found by Optuna
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Initialize XGBoost model with best hyperparameters
best_model = XGBRegressor(n_jobs=-1, objective='reg:squaredlogerror', **best_params, random_state=42)

# Train XGBoost model with full training data
best_model.fit(X, y)

# Make predictions on test data
prediction = best_model.predict(test.drop('id', axis=1))

# Prepare predictions for submission
submission = pd.DataFrame({'id': test['id'], 'rings': prediction})
submission.to_csv('C:/Users/BHARAT/Desktop/abalone.csv', index=False)

# Evaluate model using cross-validation and print mean squared log error
cv = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_log_error', n_jobs=-1)
print(cv.round(2), '\n Mean squared log error: ', round(-cv.mean(), 5))
