import os
import io
import pytest
import json
import datetime
from app import app, db, allowed_file
from models import User, File, ForecastResult
from flask import url_for

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()

# ======================
# Тесты утилит
# ======================

def test_allowed_file():
    assert allowed_file('data.csv')
    assert allowed_file('report.xlsx')
    assert allowed_file('DATA.CSV')
    assert not allowed_file('data.txt')
    assert not allowed_file('nodot')
    assert not allowed_file('archive.tar.gz')

# ======================
# Тесты авторизации
# ======================

def test_get_auth_pages(client):
    assert client.get('/register').status_code == 200
    assert client.get('/login').status_code == 200

def test_register_and_login(client):
    resp = client.post('/register', data={'email':'test@example.com','password':'1234'})
    assert resp.status_code == 302
    resp2 = client.post('/register', data={'email':'test@example.com','password':'1234'}, follow_redirects=True)
    assert 'Email уже зарегистрирован' in resp2.get_data(as_text=True)
    resp3 = client.post('/login', data={'email':'wrong@example.com','password':'1234'}, follow_redirects=True)
    assert 'Неверный email или пароль' in resp3.get_data(as_text=True)
    resp4 = client.post('/login', data={'email':'test@example.com','password':'1234'})
    assert resp4.status_code == 302

def test_logout_requires_login(client):
    resp = client.get('/logout', follow_redirects=False)
    assert resp.status_code == 302 and '/login' in resp.headers['Location']

# ======================
# Тесты загрузки файла
# ======================

def test_upload_errors(client):
    client.post('/register', data={'email':'user@example.com','password':'pass'})
    client.post('/login', data={'email':'user@example.com','password':'pass'})

    resp = client.post('/upload', data={}, content_type='multipart/form-data', follow_redirects=True)
    assert 'Файл не прикреплён' in resp.get_data(as_text=True)

    data_stream = io.BytesIO(b'data')
    data_stream.name = ''
    resp2 = client.post('/upload', data={'file':(data_stream,'')}, content_type='multipart/form-data', follow_redirects=True)
    assert 'Имя файла пустое' in resp2.get_data(as_text=True)

    bad_stream = io.BytesIO(b'data')
    bad_stream.name = 'bad.txt'
    resp3 = client.post('/upload', data={'file':(bad_stream,'bad.txt')}, content_type='multipart/form-data', follow_redirects=True)
    assert 'Недопустимый формат' in resp3.get_data(as_text=True)

    empty = io.BytesIO(b'')
    empty.name = 'empty.csv'
    resp4 = client.post('/upload', data={'file':(empty,'empty.csv')}, content_type='multipart/form-data', follow_redirects=True)
    assert 'Файл пустой' in resp4.get_data(as_text=True)

    badcsv = io.BytesIO(b'not,a,csv')
    badcsv.name = 'test.csv'
    resp5 = client.post('/upload', data={'file':(badcsv,'test.csv')}, content_type='multipart/form-data', follow_redirects=True)
    assert 'Ошибка при сохранении файла' in resp5.get_data(as_text=True)

# Основной сценарий успешной загрузки

def test_file_workflow(client):
    client.post('/register', data={'email':'user2@example.com','password':'pass'})
    client.post('/login', data={'email':'user2@example.com','password':'pass'})
    csv_content = 'category,date,value\nA,2021-01-01,10\nA,2021-01-02,20'
    stream = io.BytesIO(csv_content.encode('utf-8'))
    stream.name = 'data.csv'
    client.post('/upload', data={'file':(stream,'data.csv')}, content_type='multipart/form-data')
    files = File.query.all()
    assert len(files) == 1

# ======================
# Тесты download/delete
# ======================

def test_download_and_delete(client):
    client.post('/register', data={'email':'u@example.com','password':'p'})
    client.post('/login', data={'email':'u@example.com','password':'p'})
    f = File(filename='f.csv', data=b'abc', mimetype='text/csv', user_id=1, categories=[], start_date=None, end_date=None)
    db.session.add(f); db.session.commit()

    resp = client.get(f'/download/{f.id}')
    assert resp.status_code == 200 and resp.data == b'abc'

    resp2 = client.get(f'/delete/{f.id}', follow_redirects=False)
    assert resp2.status_code == 302 and File.query.get(f.id) is None


def test_download_nonexistent(client):
    client.post('/register', data={'email':'a@b.com','password':'p'})
    client.post('/login', data={'email':'a@b.com','password':'p'})
    resp = client.get('/download/999', follow_redirects=False)
    assert resp.status_code == 404


def test_delete_nonexistent(client):
    client.post('/register', data={'email':'a2@b.com','password':'p'})
    client.post('/login', data={'email':'a2@b.com','password':'p'})
    resp = client.get('/delete/999', follow_redirects=False)
    assert resp.status_code == 404

# ======================
# Тесты отчетов и рекомендаций
# ======================

def test_generate_report_and_recommendations(client):
    client.post('/register', data={'email':'r@r.com','password':'p'})
    client.post('/login', data={'email':'r@r.com','password':'p'})

    resp = client.get('/generate-report', query_string={'type':'stats','file_id':1})
    assert resp.status_code == 400 and resp.json['error'] == 'Нет загруженных данных'

    csv1 = 'category,date,value\nA,2021-01-01,5\nA,2021-01-02,7'
    f1 = File(filename='f1.csv', data=csv1.encode(), mimetype='text/csv', user_id=1, categories=[], start_date=None, end_date=None)
    db.session.add(f1); db.session.commit()
    r1 = client.get('/generate-report', query_string={'type':'stats','file_id':f1.id})
    assert r1.status_code == 200 and 'count' in r1.json['text']

    r2 = client.get('/generate-report', query_string={'type':'sales','file_id':f1.id})
    assert r2.status_code == 200 and 'plot' in r2.json

    csv2 = 'category,amount\nX,3\nY,4'
    f2 = File(filename='f2.csv', data=csv2.encode(), mimetype='text/csv', user_id=1, categories=[], start_date=None, end_date=None)
    db.session.add(f2); db.session.commit()
    r3 = client.get('/generate-report', query_string={'type':'demand','file_id':f2.id})
    assert r3.status_code == 200 and r3.json['plot']['data'][0]['x'] == ['X','Y']

    fr = ForecastResult(user_id=1, file_id=f1.id, model_type='test', report_data={'foo':'bar'})
    db.session.add(fr); db.session.commit()

    rd = client.get('/report-data', query_string={'report_id':fr.id})
    assert rd.status_code == 200 and rd.json['foo'] == 'bar'

    rec_missing = client.get('/recommendations')
    assert rec_missing.json['error'] == 'report_id не задан'
    rec_not_found = client.get('/recommendations', query_string={'report_id':999})
    assert rec_not_found.json['error'] == 'Отчёт не найден'
    rec_ok = client.get('/recommendations', query_string={'report_id':fr.id})
    assert rec_ok.json['forecast'] == {} and rec_ok.json['historical'] == {} and rec_ok.json['insights'] == []

# ======================
# Тест для polynomial модели
# ======================

def test_forecast_polynomial(client):
    client.post('/register', data={'email':'pol@pol.com','password':'p'})
    client.post('/login', data={'email':'pol@pol.com','password':'p'})
    csv_data = 'category,date,value\nA,2021-01-01,1\nA,2021-01-02,2'
    start = datetime.date(2021, 1, 1)
    end = datetime.date(2021, 1, 2)
    file = File(
        filename='pol.csv',
        data=csv_data.encode(),
        mimetype='text/csv',
        user_id=1,
        categories=['A'],
        start_date=start,
        end_date=end
    )
    db.session.add(file)
    db.session.commit()
    payload = {
        'file_id': file.id,
        'category': 'A',
        'model': 'polynomial',
        'degree': 2,
        'include_bias': False,
        'future_days': 3
    }
    resp = client.post('/forecast', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'historical' in data and 'forecast' in data and 'insights' in data
    assert data['forecast']['dates'] == ['2021-01-03', '2021-01-04', '2021-01-05']
    assert len(data['forecast']['yhat']) == 3

# ======================
# Тест для arima модели
# ======================

def test_forecast_arima(client):
    client.post('/register', data={'email':'ari@ari.com','password':'p'})
    client.post('/login', data={'email':'ari@ari.com','password':'p'})
    dates = [f'2021-01-{str(d).zfill(2)}' for d in range(1, 11)]
    csv_lines = ['category,date,value'] + [f"A,{dates[i]},{i+1}" for i in range(10)]
    csv_data = '\n'.join(csv_lines)
    file = File(
        filename='ari.csv',
        data=csv_data.encode(),
        mimetype='text/csv',
        user_id=1,
        categories=['A'],
        start_date=datetime.date(2021,1,1),
        end_date=datetime.date(2021,1,10)
    )
    db.session.add(file)
    db.session.commit()
    payload = {'file_id': file.id, 'category': 'A', 'model': 'arima', 'm': 1, 'future_days': 3}
    resp = client.post('/forecast', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'historical' in data and 'forecast' in data and 'insights' in data
    assert len(data['forecast']['dates']) == 3

# ======================
# Тест для lstm модели
# ======================

def test_forecast_lstm(client):
    client.post('/register', data={'email':'lstm@lstm.com','password':'p'})
    client.post('/login', data={'email':'lstm@lstm.com','password':'p'})
    dates = [f'2021-01-{str(d).zfill(2)}' for d in range(1, 11)]
    csv_lines = ['category,date,value'] + [f"A,{dates[i]},{i+1}" for i in range(10)]
    csv_data = '\n'.join(csv_lines)
    file = File(
        filename='lstm.csv',
        data=csv_data.encode(),
        mimetype='text/csv',
        user_id=1,
        categories=['A'],
        start_date=datetime.date(2021,1,1),
        end_date=datetime.date(2021,1,10)
    )
    db.session.add(file)
    db.session.commit()
    payload = {
        'file_id': file.id,
        'category': 'A',
        'model': 'lstm',
        'look_back': 3,
        'predict_range': 2,
        'units1': 2,
        'units2': 1,
        'dropout': 0.0,
        'recurrent_dropout': 0.0,
        'epoch_const': 2,
        'learning_rate': 0.01
    }
    resp = client.post('/forecast', json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'historical' in data and 'forecast' in data and 'insights' in data
    assert len(data['forecast']['yhat']) == 2
