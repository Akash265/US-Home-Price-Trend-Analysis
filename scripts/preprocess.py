import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import PowerTransformer

class FeaturePreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.power_transformer = PowerTransformer(method='yeo-johnson')
        self.selected_features = None
        self.categorical_features = None
        
    def calculate_vif(self, X):
        """Calculate VIF for each feature"""
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
    
    def select_features(self, X, vif_threshold=10000):
        """Select features based on VIF threshold"""
        features_to_keep = []
        # Always keep Interest_Rates and Unemployment_Rate as they show low multicollinearity
        essential_features = ['Interest_Rates', 'Unemployment_Rate']
        
        # Add essential features if they exist in the dataset
        features_to_keep.extend([f for f in essential_features if f in X.columns])
        
        # Calculate VIF for remaining features
        remaining_features = [col for col in X.columns if col not in essential_features]
        if remaining_features:
            temp_X = X[remaining_features].copy()
            while True:
                vif_data = self.calculate_vif(temp_X)
                if vif_data['VIF'].max() <= vif_threshold:
                    features_to_keep.extend(temp_X.columns.tolist())
                    break
                worst_feature = vif_data.loc[vif_data['VIF'].idxmax(), 'Feature']
                temp_X.drop(worst_feature, axis=1, inplace=True)
                
        self.selected_features = features_to_keep
        return features_to_keep
    
    def create_interaction_terms(self, X):
        """Create interaction terms for strongly correlated variables"""
        # Based on correlation analysis results
        strong_correlations = [
            ('Per_Capita_GDP', 'Construction_Price_Index'),
            ('Housing_Subsidies', 'Consumer_Price_Index'),
            ('Total_Households', 'Median_Household_Income'),
            ('Working_Population', 'Per_Capita_GDP')
        ]
        
        for var1, var2 in strong_correlations:
            if var1 in X.columns and var2 in X.columns:
                X[f'{var1}_{var2}_interaction'] = X[var1] * X[var2]
        
        return X
    
    def create_polynomial_features(self, X, degree=2):
        """Create polynomial features for non-linear relationships"""
        for column in X.columns:
            if column not in self.categorical_features:
                X[f'{column}_squared'] = X[column]**degree
        return X
    
    def get_categorical(self, X):
        categorical_feats = []
        for col in X.columns:
            if X[col].nunique()/len(X[col]) < 0.1:
                categorical_feats.append(col)
        self.categorical_features = categorical_feats
        return categorical_feats
    
    def fit_transform(self, X, y):
        """Complete preprocessing pipeline"""
        # 1. Feature Selection
        selected_features = self.select_features(X)
        X_selected = X[selected_features].copy()
        categorical_features = self.get_categorical(X_selected) 
        # 2. Feature Engineering
        X_engineered = self.create_interaction_terms(X_selected)
        X_engineered = self.create_polynomial_features(X_engineered)
        
        num_features = [col for col in X_engineered.columns if col not in categorical_features]
        # 3. Data Transformation
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_engineered[num_features]),
            columns=num_features,
            index=X_engineered.index
        )
        for cat_col in categorical_features:      
            X_scaled[cat_col] = X_engineered[cat_col]
        # Transform target variable (handling non-normality)
        y_transformed = pd.Series(
            self.power_transformer.fit_transform(y.values.reshape(-1, 1)).ravel(),
            index=y.index
        )
        
        return X_scaled, y_transformed
    
    def transform(self, X, y):
        """Transform new data using fitted preprocessor"""
        if self.selected_features is None:
            raise ValueError("Preprocessor must be fitted before transform")
            
        X_selected = X[self.selected_features].copy()
        X_engineered = self.create_interaction_terms(X_selected)
        X_engineered = self.create_polynomial_features(X_engineered)
        
        num_features = [col for col in X_engineered.columns if col not in self.categorical_features]
        X_scaled = pd.DataFrame(
            self.scaler.transform(X_engineered[num_features]),
            columns=num_features,
            index=X_engineered.index
        )
        for cat_col in self.categorical_features:      
            X_scaled[cat_col] = X_engineered[cat_col]
            
        y_scaled = pd.Series(
            self.power_transformer.transform(y.values.reshape(-1, 1)).ravel(),
            index=y.index
        )
        
        return X_scaled, y_scaled
    
    def get_scaler(self):
        return self.scaler
    

