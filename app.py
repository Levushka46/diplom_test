from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from models import ForecastResult, db, User, File
from datetime import datetime
import pandas as pd
import plotly.express as px
from io import BytesIO, StringIO
import os
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from prophet import Prophet


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

        return jsonify({'message': 'Файл успешно загружен'})

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
            fig = px.line(df, x='Дата', y='Продажи', title='Динамика продаж')
            return jsonify({'plot': fig.to_dict()})
        
        elif report_type == 'demand':
            fig = px.bar(df, x='Товар', y='Спрос', title='Спрос на товары')
            return jsonify({'plot': fig.to_dict()})
            
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
    return render_template('dashboard.html', files=files)

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    try:
        # Получаем данные из запроса
        file_id = request.json.get('file_id')
        category = request.json.get('category')
        model = request.json.get('model')
        
        # Получаем файл из БД
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': 'File not found'}), 404
        
        # Читаем CSV
        df = pd.read_csv(BytesIO(file.data))
        df = df[df['category'] == category]
        
        if model == 'prophet':
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

        elif model == 'arima':
            # 1. Загрузка данных и подготовка
            # Предполагается, что у вас есть DataFrame df с колонками 'date' и 'value'
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').asfreq('D')   # задаём дневную частоту (при необходимости менять)
            df['value'].interpolate(method='time', inplace=True)  # заполняем пропуски

            # 2. Разделение на train и test (опционально)
            #train = df.iloc[:-30]   # всё, кроме последних 30 дней
            #test  = df.iloc[-30:]   # последние 30 дней для валидации
            train = df

            # 3. Подбор гиперпараметров (p, d, q) автоматически
            stepwise_model = auto_arima(train['value'], start_p=0, start_q=0,
                                        max_p=5, max_q=5, m=7,          # m=7 — для недельной сезонности
                                        start_P=0, seasonal=True,       # если есть сезонность
                                        d=None, D=1, trace=True,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=True)

            print(stepwise_model.summary())

            # 4. Обучение финальной модели ARIMA
            # Если вы хотите взять параметры из auto_arima:
            order = stepwise_model.order     # (p, d, q)
            seasonal_order = stepwise_model.seasonal_order  # (P, D, Q, m)

            model = ARIMA(train['value'], order=order, seasonal_order=seasonal_order)
            fitted = model.fit()
            print(fitted.summary())

            # 5. Прогноз на будущие дни
            n_periods = 30
            forecast_result = fitted.get_forecast(steps=n_periods)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            # 6. Визуализация
            plt.figure(figsize=(12,6))
            plt.plot(train.index, train['value'], label='Train')
            plt.plot(test.index, test['value'], label='Test', color='orange')
            plt.plot(forecast.index, forecast, label='Forecast', color='green')
            plt.fill_between(forecast.index,
                            conf_int.iloc[:, 0],
                            conf_int.iloc[:, 1], color='lightgreen', alpha=0.5)
            plt.legend()
            plt.title('ARIMA Forecast of Sales')
            plt.xlabel('Date')
            plt.ylabel('Sales')
            plt.show()
        
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
        
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f'Error in forecast: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)