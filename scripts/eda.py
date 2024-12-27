import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import statsmodels.api as sm

def load_and_prepare_data():
    # Assuming data is in a CSV file named 'prepared_dataset.csv'
    df = pd.read_csv('dataset.csv')
    df.set_index('date', inplace=True)
    return df.drop(['Year', 'Month'], axis=1)

def check_linearity(df):
    """Check linearity between target and predictors"""
    target = 'Home_Price_Index'
    predictors = df.columns.drop(target)
    
    plt.figure(figsize=(15, 10))
    for i, predictor in enumerate(predictors, 1):
        plt.subplot(3, 4, i)
        plt.scatter(df[predictor], df[target], alpha=0.5)
        plt.xlabel(predictor)
        plt.ylabel('Home Price Index')
        z = np.polyfit(df[predictor], df[target], 1)
        p = np.poly1d(z)
        plt.plot(df[predictor], p(df[predictor]), "r--", alpha=0.8)
    plt.tight_layout()
    plt.show()

def check_multicollinearity(df):
    """Calculate VIF for each predictor"""
    X = df.drop('Home_Price_Index', axis=1)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def check_normality_and_outliers(df):
    """Check normality of target variable and identify outliers"""
    target = df['Home_Price_Index']
    
    # Normality test
    stat, p_value, _, _ = jarque_bera(target)
    
    # Plot distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    sns.histplot(target, kde=True)
    plt.title('Distribution of Home Price Index')
    
    # Q-Q plot
    plt.subplot(122)
    stats.probplot(target, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Home Price Index')
    plt.tight_layout()
    plt.show()
    
    return stat, p_value

def check_heteroscedasticity(df):
    """Test for heteroscedasticity using Breusch-Pagan test"""
    X = sm.add_constant(df.drop('Home_Price_Index', axis=1))
    y = df['Home_Price_Index']
    model = sm.OLS(y, X).fit()
    
    _, p_value, _, _ = het_breuschpagan(model.resid, model.model.exog)
    
    # Plot residuals
    plt.figure(figsize=(10, 5))
    plt.scatter(model.fittedvalues, model.resid)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Values')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
    
    return p_value

def analyze_correlations(df):
    """Analyze correlations between variables"""
    corr_matrix = df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
    
    return corr_matrix['Home_Price_Index'].sort_values(ascending=False)

def main():
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    print("1. Correlation Analysis:")
    correlations = analyze_correlations(df)
    print("\nCorrelations with Home Price Index:")
    print(correlations)
    
    print("\n2. Multicollinearity Check:")
    vif_results = check_multicollinearity(df)
    print("\nVariance Inflation Factors:")
    print(vif_results)
    
    print("\n3. Normality Test Results:")
    jb_stat, jb_p_value = check_normality_and_outliers(df)
    print(f"Jarque-Bera statistic: {jb_stat:.4f}")
    print(f"p-value: {jb_p_value:.4f}")
    
    print("\n4. Heteroscedasticity Test Results:")
    bp_p_value = check_heteroscedasticity(df)
    print(f"Breusch-Pagan test p-value: {bp_p_value:.4f}")
    
    print("\n5. Checking linearity relationships...")
    check_linearity(df)
    

if __name__ == "__main__":
    main()