# US-Home-Price-Trend-Analysis


This project aims to predict the Home Price Index using various economic indicators. The project includes data fetching, preprocessing, exploratory data analysis, and machine learning modeling using XGBoost.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Data Fetching](#data-fetching)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)

## Overview

The Home Price Index (HPI) is a crucial economic indicator that reflects changes in housing prices over time. This project uses data from the Federal Reserve Economic Data (FRED) API to fetch relevant economic indicators and build a predictive model using XGBoost. The project includes the following steps:

1. **Data Fetching**: Retrieve economic data from the FRED API.
2. **Data Preprocessing**: Clean and preprocess the data, including feature selection, engineering, and transformation.
3. **Exploratory Data Analysis (EDA)**: Perform EDA to understand the data and check assumptions.
4. **Model Training**: Train an XGBoost model to predict the Home Price Index.
5. **Model Evaluation**: Evaluate the model's performance using various metrics and interpret the results.

## Features

- **Data Fetching**: Automate the process of fetching data from the FRED API.
- **Data Preprocessing**: Implement feature selection, interaction terms, polynomial features, and scaling.
- **Exploratory Data Analysis**: Conduct linearity checks, multicollinearity checks, normality tests, and heteroscedasticity tests.
- **Model Training**: Train an XGBoost model with cross-validation.
- **Model Evaluation**: Evaluate the model using RMSE, MAE, and R² metrics. Generate feature importance plots and SHAP values for interpretation.

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - `requests`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scipy`
  - `statsmodels`
  - `xgboost`
  - `shap`
  - `dotenv`
  - `scikit-learn`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/home-price-index-prediction.git
   cd home-price-index-prediction
   ```
2. Install the required Python packages:
   
  ```bash
  pip install -r requirements.txt
  ```
3. Create a .env file in the scripts directory with your FRED API key:
   
  ```bash
  FRED_API_KEY=your_fred_api_key
  ```
## Data Fetching
The fetch_and_save_data.py script fetches data from the FRED API, processes it, and saves it to a CSV file. The script uses the get_params, get_data, and fetch_and_save_data functions to handle the API requests and data processing.

## Data Preprocessing
The FeaturePreprocessor class in preprocessor.py handles the preprocessing steps, including feature selection, interaction term creation, polynomial feature generation, and data scaling. The fit_transform method applies the entire preprocessing pipeline to the data.

## Exploratory Data Analysis
The exploratory_data_analysis.py script performs various EDA tasks, such as checking linearity, multicollinearity, normality, and heteroscedasticity. It also generates correlation heatmaps and other visualizations.

## Model Training and Evaluation
The train_and_evaluate_model.py script trains an XGBoost model using the preprocessed data. It includes cross-validation, model training, and evaluation. The script also generates feature importance plots and SHAP values for model interpretation.
