import pandas as pd
import numpy as np
import os

def generate_sample_data_v1():
    """生成版本1的示例数据"""
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'area': np.random.normal(1500, 500, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'floors': np.random.randint(1, 4, n_samples),
        'year_built': np.random.randint(1950, 2023, n_samples),
        'price': 0  # 将在下面计算
    })
    
    # 计算价格（基础公式 + 噪声）
    data['price'] = (
        data['area'] * 100 + 
        data['bedrooms'] * 50000 + 
        data['floors'] * 20000 + 
        (data['year_built'] - 1950) * 1000 +
        np.random.normal(0, 50000, n_samples)
    )
    
    # 保存数据
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/training_data.csv', index=False)
    print("Generated sample data v1")

def generate_sample_data_v2():
    """生成版本2的改进数据"""
    np.random.seed(123)
    n_samples = 1500  # 更多样本
    
    data = pd.DataFrame({
        'area': np.random.normal(1600, 400, n_samples).astype(int),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'floors': np.random.randint(1, 3, n_samples),
        'year_built': np.random.randint(1970, 2023, n_samples),
        'price': 0
    })
    
    # 改进的价格公式
    data['price'] = (
        data['area'] * 120 + 
        data['bedrooms'] * 45000 + 
        data['floors'] * 25000 + 
        (data['year_built'] - 1970) * 1200 +
        np.random.normal(0, 40000, n_samples)  # 减少噪声
    )
    
    # 移除异常值
    data = data[(data['price'] > 50000) & (data['price'] < 2000000)]
    
    os.makedirs('data/processed', exist_ok=True)
    data.to_csv('data/processed/training_data.csv', index=False)
    print("Generated improved sample data v2")

if __name__ == '__main__':
    generate_sample_data_v1()