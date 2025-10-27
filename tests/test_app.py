import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.main import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """测试首页端点"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'healthy' in response.data

def test_health_endpoint(client):
    """测试健康检查端点"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json['status'] == 'healthy'

def test_predict_endpoint(client):
    """测试预测端点"""
    test_data = {
        'area': 1500,
        'bedrooms': 3,
        'floors': 2,
        'year_built': 2010
    }
    
    response = client.post('/predict', json=test_data)
    assert response.status_code == 200
    assert 'prediction' in response.json

def test_predict_invalid_data(client):
    """测试无效数据预测"""
    test_data = {'invalid': 'data'}
    response = client.post('/predict', json=test_data)
    assert response.status_code == 400