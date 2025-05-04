from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from models import ForecastResult, db, User, File
from datetime import datetime
import pandas as pd
import plotly.express as px
from io import BytesIO, StringIO
import os
from sqlalchemy.orm import joinedload
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.urandom(24),
    SQLALCHEMY_DATABASE_URI='sqlite:///sales_analytics.db',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    UPLOAD_FOLDER='static/uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16 MB
)

# Инициализация расширений
db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Создание таблиц
with app.app_context():
    db.create_all()

# ========================
# Маршруты аутентификации
# ========================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Неверный email или пароль!', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email уже зарегистрирован!', 'warning')
            return redirect(url_for('register'))
        
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Регистрация успешна! Войдите в систему.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ========================
# Маршруты работы с данными
# ========================
@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не выбран'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Неверное имя файла'}), 400

    try:
        # Важно: сохраняем содержимое файла в переменную
        file_data = file.read()
        
        # Проверка что файл не пустой
        if len(file_data) == 0:
            return jsonify({'error': 'Файл пуст'}), 400
            
        # Чтение для получения метаданных
        df = pd.read_csv(BytesIO(file_data))
        
        # Сохранение в БД с исходными данными
        new_file = File(
            filename=secure_filename(file.filename),
            data=file_data,  # Используем сохраненные данные
            mimetype=file.mimetype,
            user_id=current_user.id,
            categories=df['category'].unique().tolist(),
            start_date=pd.to_datetime(df['date']).min(),
            end_date=pd.to_datetime(df['date']).max()
        )
        db.session.add(new_file)
        db.session.commit()

        return redirect(url_for('dashboard'))

    except Exception as e:
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500

@app.route('/data')
@login_required
def data():
    files = File.query.filter_by(user_id=current_user.id).all()
    return render_template('data.html', files=files)

@app.route('/download/<int:file_id>')
@login_required
def download_file(file_id):
    file = File.query.get_or_404(file_id)
    return Response(
        file.data,
        mimetype=file.mimetype,
        headers={'Content-Disposition': f'attachment; filename={file.filename}'}
    )

@app.route('/delete/<int:file_id>')
@login_required
def delete_file(file_id):
    file = File.query.get_or_404(file_id)
    db.session.delete(file)
    db.session.commit()
    flash('Файл удален!', 'success')
    return redirect(url_for('data'))

# ========================
# Маршруты отчетов
# ========================
@app.route('/generate-report')
@login_required
def generate_report():
    report_type = request.args.get('type', 'stats')
    
    try:
        file = File.query.filter_by(user_id=current_user.id).order_by(File.uploaded_at.desc()).first()
        if not file:
            return jsonify({'error': 'Нет загруженных данных'}), 400
        
        df = pd.read_csv(StringIO(file.data.decode('utf-8')))
        
        if report_type == 'stats':
            report = df.describe().to_string()
            return jsonify({'text': report})
        
        elif report_type == 'sales':
            fig = px.line(df, x='date', y='value', title='Динамика продаж')
            fig_dict = fig.to_dict()
            fig_dict['data'][0]['x'] = list(fig_dict['data'][0]['x'])
            return jsonify({'plot': fig_dict})
        
        elif report_type == 'demand':
            fig = px.bar(df, x='category', y='value', title='Спрос на товары')
            fig_dict = fig.to_dict()
            fig_dict['data'][0]['x'] = list(fig_dict['data'][0]['x'])
            return jsonify({'plot': fig_dict})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ========================
# Основные маршруты
# ========================
@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
@login_required
def dashboard():
    files = File.query.filter_by(user_id=current_user.id).all()
    reports = ForecastResult.query.options(joinedload(ForecastResult.file)).filter_by(user_id=current_user.id).all()
    return render_template('dashboard.html', files=files, reports=reports)

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    try:
        # Получаем данные из запроса
        file_id = request.json.get('file_id')
        category = request.json.get('category')
        model_type = request.json.get('model')
        
        # Получаем файл из БД
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': 'File not found'}), 404
        
        # Читаем CSV
        df = pd.read_csv(BytesIO(file.data))
        df = df[df['category'] == category]
        
        if model_type == 'prophet':
            # Готовим данные для Prophet
            df = df.rename(columns={'date': 'ds', 'value': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])

            # Создаем и обучаем модель
            m = Prophet()
            m.fit(df)
            
            # Создаем будущие даты
            future = m.make_future_dataframe(periods=30)
            
            # Прогнозируем
            forecast = m.predict(future)

        elif model_type == 'arima':
            # 1. Загрузка и подготовка данных
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').asfreq('D')
            df['value'].interpolate(method='time', inplace=True)

            # Переименуем для единообразия с Prophet
            df = df.rename(columns={'value': 'y'})
            df = df.reset_index().rename(columns={'date': 'ds'})
            df['ds'] = pd.to_datetime(df['ds'])

            # 2. Разделим на train / test (по желанию)
            #train = df.iloc[:-30]
            #test  = df.iloc[-30:]
            train = df

            # 3. Подбор параметров SARIMA
            stepwise_model = auto_arima(
                train['y'],
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=7,               # недельная сезонность
                seasonal=True,
                d=None, D=1,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            # 4. Обучение финальной модели
            order = stepwise_model.order
            seasonal_order = stepwise_model.seasonal_order
            model = ARIMA(train['y'], order=order, seasonal_order=seasonal_order)
            fitted = model.fit()

            # 5. Прогноз на будущие 30 дней
            n_periods = 30
            forecast_res = fitted.get_forecast(steps=n_periods)
            forecast_mean = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int(alpha=0.05)

            # 6. Формируем DataFrame для прогноза
            last_date = df['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=n_periods, freq='D')

            forecast = pd.DataFrame({
                'ds': future_dates,
                'yhat': forecast_mean.values,
                'yhat_lower': conf_int.iloc[:, 0].values,
                'yhat_upper': conf_int.iloc[:, 1].values
            })

        elif model_type == 'lstm':
            # 1. Загрузка и подготовка данных
            # Предполагается, что у вас есть DataFrame df с колонками 'date' и 'value'
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').asfreq('D')
            df['value'].interpolate(method='time', inplace=True)

            # Переименуем столбцы для единообразия
            df = df.rename(columns={'value': 'y'}).reset_index().rename(columns={'date': 'ds'})

            # 2. Масштабирование
            scaler = MinMaxScaler(feature_range=(0, 1))
            y_scaled = scaler.fit_transform(df['y'].values.reshape(-1, 1))

            # 3. Создание обучающих последовательностей
            def create_sequences(data, look_back=14):
                X, y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:(i + look_back), 0])
                    y.append(data[i + look_back, 0])
                return np.array(X), np.array(y)

            LOOK_BACK = 7  # число дней для входной последовательности
            X, y = create_sequences(y_scaled, look_back=LOOK_BACK)

            # Разделим на train и test (оставим последние 30 точек для проверки)
            #train_size = len(X) - 30
            #X_train, X_test = X[:train_size], X[train_size:]
            #y_train, y_test = y[:train_size], y[train_size:]

            # Приведём входы к форме [samples, timesteps, features]
            #X_train = X_train.reshape((X_train.shape[0], LOOK_BACK, 1))
            #X_test  = X_test.reshape((X_test.shape[0], LOOK_BACK, 1))

            X_train = X.reshape((X.shape[0], LOOK_BACK, 1))
            y_train = y

            # 4. Построение LSTM-модели
            model = Sequential([
                LSTM(50, activation='tanh', input_shape=(LOOK_BACK, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')

            # 5. Обучение
            model.fit(X_train, y_train, epochs=20, batch_size=16)

            # 6. Прогноз на следующие 30 дней
            # Для итеративного прогноза будем подставлять каждый новый прогноз в конец последовательности
            last_sequence = y_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
            forecast_scaled = []
            for _ in range(30):
                pred = model.predict(last_sequence)
                forecast_scaled.append(pred[0,0])
                # обновляем sequence
                last_sequence = np.append(last_sequence[:,1:,:], [pred], axis=1)

            # Отмасштабируем обратно
            forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1)).flatten()

            # 7. Формируем даты прогноза
            last_date = df['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=30, freq='D')

            # 8. Собираем итоговый словарь в том же формате
            result = {
                'historical': {
                    'dates':  df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': df['y'].tolist()
                },
                'forecast': {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    #'yhat': list(np.round(forecast, 2))
                    'yhat': list(map(lambda x: round(float(x), 2), forecast)),
                }
            }
            save_forecast_result(file_id, model_type, result)
            return jsonify(result)
        
        # Форматируем результат
        result = {
            'historical': {
                'dates': df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'values': df['y'].tolist()
            },
            'forecast': {
                'dates': forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'yhat': forecast['yhat'].round(2).tolist(),
                'yhat_lower': forecast['yhat_lower'].round(2).tolist(),
                'yhat_upper': forecast['yhat_upper'].round(2).tolist()
            }
        }
        save_forecast_result(file_id, model_type, result)
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'Error in forecast: {str(e)}')
        return jsonify({'error': str(e)}), 500

def save_forecast_result(file_id, model_type, report_data):
    try:
        forecast_result = ForecastResult(
            user_id=current_user.id,
            file_id=file_id,
            model_type=model_type,
            report_data=report_data
        )
        db.session.add(forecast_result)
        db.session.commit()
    except Exception as e:
        app.logger.error(f'Error saving forecast result: {str(e)}')
        db.session.rollback()


@app.route('/report-data', methods=['GET'])
def report_data():
    # 1. Получаем report_id из query string
    report_id = request.args.get('report_id')
    if not report_id:
        return jsonify({'error': 'report_id is required'}), 400

    # 2. Пытаемся найти запись в БД
    try:
        report_id = int(report_id)
    except ValueError:
        return jsonify({'error': 'report_id must be an integer'}), 400

    report = ForecastResult.query.get(report_id)
    if not report:
        return jsonify({'error': 'Report not found'}), 404

    # 3. Достаём сохранённый JSON из модели.
    #    Предполагаем, что у вас есть поле `result` типа JSON или Text
    #    со структурой {'historical': {...}, 'forecast': {...}}
    result = report.report_data
    # Если у вас это строка, а не JSONB/JSONField, можете раскомментировать:
    # result = json.loads(report.result)

    # 4. Отдаём напрямую клиенту
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)