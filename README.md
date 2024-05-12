# Abalone Age Prediction using Machine Learning

This repository contains code for predicting the age (number of rings) of abalone using machine learning models. Abalone are a type of marine mollusk with shells that are commonly used in various culinary dishes. The age of an abalone is typically determined by counting the number of rings on its shell.

## Problem Statement

The goal of this project is to build machine learning models that can accurately predict the age of abalone based on physical measurements and other attributes. This prediction task is treated as a regression problem where the target variable is the number of rings (indicative of age).

## Dataset

The dataset used for training and evaluation contains features such as:
- Length of the abalone shell (`length`)
- Diameter of the abalone shell (`diameter`)
- Height of the abalone (`height`)
- Sex of the abalone (`sex`: encoded as 0 for female, 1 for male, and 2 for infant)
- Other derived features like `height weight` based on shell weight and height

The dataset is split into a training set and a test set, and models are trained on the training data to predict the age of abalone in the test set.

## Models Used

The following machine learning models are explored and optimized:
- **XGBoost (Extreme Gradient Boosting)**
- **LightGBM (Light Gradient Boosting Machine)**
- **CatBoost (Categorical Boosting)**

Hyperparameters of these models are optimized using the Optuna library to achieve the best performance.

## Repository Structure

- **`abalone_prediction.py`**: Python script containing the main code for data preprocessing, model training, and prediction.
- **`hyperparameters.py`**: Python script defining hyperparameter search space for model optimization.
- **`train.csv`**: Training dataset containing features and target variable (`rings`).
- **`test.csv`**: Test dataset for making predictions.
