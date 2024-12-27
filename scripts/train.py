import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
import shap

class ModelTrainer:
    def __init__(self, params=None):
        # Default XGBoost parameters based on the data characteristics
        self.params = params if params else {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 3,
            'min_child_weight': 4,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'objective': 'reg:squarederror',
            'random_state': 42
        }
        self.model = XGBRegressor(**self.params)
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional early stopping"""
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_metric='rmse',
            early_stopping_rounds=50,
            verbose=100
        )
        
    def cross_validate(self, X, y, n_splits=5):
        """Perform cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kf, scoring='r2')
        return scores
    
    def evaluate(self, X, y_true):
        """Calculate performance metrics"""
        y_pred = self.model.predict(X)
        
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
        
        return metrics, y_pred
    
    def plot_feature_importance(self, feature_names):
        """Plot feature importance"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
    def analyze_shap_values(self, X):
        """Generate SHAP values for model interpretation"""
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        shap.summary_plot(shap_values, X)
        
        return shap_values, explainer

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """Complete training and evaluation pipeline"""
    # Initialize and train model
    trainer = ModelTrainer()
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = trainer.cross_validate(X_train, y_train)
    print(f"Cross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    test_metrics, y_pred = trainer.evaluate(X_test, y_test)
    print("\nTest Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Feature importance
    print("\nGenerating feature importance plot...")
    trainer.plot_feature_importance(feature_names)
    
    # SHAP analysis
    print("\nGenerating SHAP analysis...")
    shap_values, explainer = trainer.analyze_shap_values(X_train)
    
    return trainer, y_pred, test_metrics, shap_values

