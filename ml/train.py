import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import mlflow
import mlflow.sklearn
import os

def load_data():
    """加载训练数据"""
    data_path = os.getenv('DATA_PATH', 'data/processed/training_data.csv')
    return pd.read_csv(data_path)

def train_model():
    # 开始MLflow实验
    mlflow.set_experiment("house_price_prediction")
    
    with mlflow.start_run():
        # 加载数据
        data = load_data()
        X = data[['area', 'bedrooms', 'floors', 'year_built']]
        y = data['price']
        
        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 记录参数
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        # 训练模型
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # 预测和评估
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 记录指标
        mlflow.log_metrics({
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2
        })
        
        # 保存模型
        model_path = "ml/registry/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # 记录模型
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_artifact(model_path)
        
        print(f"Training completed. MSE: {mse:.2f}, R2: {r2:.2f}")
        
        return model

if __name__ == '__main__':
    train_model()