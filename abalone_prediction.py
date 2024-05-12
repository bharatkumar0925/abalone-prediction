from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import r2_score, adjusted_rand_score, mean_squared_error, mean_squared_log_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, StackingRegressor
import pandas as pd
import numpy as np
import warnings

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

def msle_loss(y_true, y_pred):
    """Custom mean squared log error (MSLE) loss function."""
    return mean_squared_log_error(y_true, y_pred)

def metrics(y_pred, y_test):
    """Function to print evaluation metrics for regression."""
    # Compute and print R2 score
    print(f'R2 score: {r2_score(y_pred, y_test)}')

    # Compute and print mean squared error (MSE)
    print(f'Mean squared error: {mean_squared_error(y_pred, y_test)}')

    # Compute and print mean squared log error (MSLE)
    print(f'Mean squared log error: {mean_squared_log_error(abs(y_pred), abs(y_test))}')

# Load train and test data from CSV files
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Convert column names to lowercase for consistency
train.columns, test.columns = train.columns.str.lower(), test.columns.str.lower()

# Map categorical variable 'sex' to numerical values
train['sex'] = train['sex'].map({'F': 0, 'M': 1, 'I': 2})
test['sex'] = test['sex'].map({'F': 0, 'M': 1, 'I': 0})

# Feature engineering: Create a new feature 'height weight' using exponential operation
train['height weight'] = train['height']**train['shell weight']
test['height weight'] = test['height']**test['shell weight']

# Drop 'id' column from the training dataset
train.drop('id', axis=1, inplace=True)

# Split data into features (X) and target (y)
X = train.drop('rings', axis=1)
y = train['rings']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define best hyperparameters for XGBoost, LightGBM, and CatBoost
best_xgb = {'n_estimators': 631, 'max_depth': 7, 'learning_rate': 0.09011309969202384, 'reg_alpha': 0.19193657111063686, 'reg_lambda': 0.5061915593882026, 'colsample_bytree': 0.8748819153424529, 'subsample': 0.7968184313773305, 'min_child_weight': 1}
best_lgbm = {'n_estimators': 950, 'max_depth': 5, 'learning_rate': 0.06412228916880802, 'reg_alpha': 17.93313806569324, 'reg_lambda': 23.007046813911384, 'colsample_bytree': 0.6559453003446982, 'subsample': 0.5361319680240149, 'min_child_weight': 5, 'num_leaves': 84}
best_cat = {'n_estimators': 943, 'max_depth': 4, 'learning_rate': 0.21937492994278393, 'reg_lambda': 1.1381210054138413, 'colsample_bylevel': 0.6232050432268481, 'subsample': 0.973037067259649}

# Initialize XGBoost, LightGBM, and CatBoost models with best hyperparameters
xgb = XGBRegressor(n_jobs=-1, random_state=42, **best_xgb, objective='reg:squaredlogerror')
lgbm = LGBMRegressor(verbose=-1, n_jobs=-1, random_state=42, **best_lgbm)
catboost = CatBoostRegressor(verbose=0, random_state=42, **best_cat)

# Define base models for StackingRegressor
base_models = [
    ('xgb', xgb),
    ('lgbm', lgbm),
    ('cat', catboost),
]

# Define middle layer models for StackingRegressor
middle_models = [
    ('lr', LinearRegression(n_jobs=-1)),
    ('sgd', SGDRegressor(penalty='elasticnet', random_state=42, learning_rate='adaptive', alpha=0.01)),
    ('mlp', MLPRegressor((50, 25), random_state=42, learning_rate='adaptive', alpha=1.0)),
]

# Define final estimator (meta learner) for StackingRegressor
final_estimator = Ridge(random_state=42, max_iter=200, alpha=10, solver='svd')

# Define StackingRegressor model with base and middle layer models
model = StackingRegressor(base_models,
                          final_estimator=StackingRegressor(middle_models, final_estimator, cv=5, n_jobs=-1),
                          cv=5, n_jobs=-1, passthrough=True)

# Define Pipeline for preprocessing and modeling
pipe = Pipeline([
    ('scaling', StandardScaler()),  # Standardize features using StandardScaler
    ('model', model)  # Apply the StackingRegressor model
])

# Fit the Pipeline on training data
pipe.fit(X_train, y_train)

# Make predictions on test data
prediction = pipe.predict(X_test).astype('int')

# Evaluate prediction using custom metrics function
metrics(prediction, y_test)

# Prepare predictions for test data and save to CSV file
id = test['id']
test = test.drop('id', axis=1)
prediction = pipe.predict(test)
temp = pd.DataFrame({'id': id, 'rings': prediction})
temp.to_csv('C:/Users/BHARAT/Desktop/abalone.csv', index=False)

# Evaluate model using cross-validation and print mean squared log error
cv = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_log_error', n_jobs=-1)
print(cv.round(2), '\n Mean squared log error: ', round(cv.mean(), 6))
