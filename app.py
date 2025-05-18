import os
from io import BytesIO, StringIO

from flask import flash, Flask, jsonify, redirect, render_template, request, Response, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from werkzeug.utils import secure_filename

import numpy as np
import pandas as pd
import plotly.express as px

from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures 
from sklearn.linear_model import LinearRegression
from sqlalchemy.orm import joinedload
from statsmodels.tsa.arima.model import ARIMA
import datetime
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from models import db, File, ForecastResult, User


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
# Разрешённые расширения
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    # 1) наличие файла
    if 'file' not in request.files:
        flash('Файл не прикреплён', 'danger')
        return redirect(url_for('dashboard'))
    
    file = request.files['file']
    if file.filename == '':
        flash('Имя файла пустое', 'danger')
        return redirect(url_for('dashboard'))
    
    # 2) проверяем расширение
    if not allowed_file(file.filename):
        flash('Недопустимый формат. Разрешено: CSV, XLSX', 'danger')
        return redirect(url_for('dashboard'))
    
    filename = secure_filename(file.filename)
    data = file.read()
    if not data:
        flash('Файл пустой', 'danger')
        return redirect(url_for('dashboard'))
    
    # 3) попытка распарсить датафрейм
    try:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext == 'csv':
            df = pd.read_csv(BytesIO(data))
        else:  # xlsx
            df = pd.read_excel(BytesIO(data))
    except Exception as e:
        flash(f'Не удалось прочитать файл: {e}', 'danger')
        return redirect(url_for('dashboard'))
    
    # 4) сохраняем в БД
    try:
        new_file = File(
            filename=filename,
            data=data,
            mimetype=file.mimetype,
            user_id=current_user.id,
            categories=df['category'].dropna().unique().tolist(),
            start_date=pd.to_datetime(df['date']).min(),
            end_date=pd.to_datetime(df['date']).max()
        )
        db.session.add(new_file)
        db.session.commit()
        flash('Файл успешно загружен', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Ошибка при сохранении файла: {e}', 'danger')
    
    return redirect(url_for('dashboard'))

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
    return redirect(url_for('dashboard'))

# ========================
# Маршруты отчетов
# ========================
@app.route('/generate-report')
@login_required
def generate_report():
    report_type = request.args.get('type', 'stats')
    file_id = request.args.get('file_id', type=int)

    try:
        # Если передан file_id — берём именно его, иначе последний
        if file_id is not None:
            file = File.query.filter_by(id=file_id, user_id=current_user.id).first()
        else:
            file = File.query \
                       .filter_by(user_id=current_user.id) \
                       .order_by(File.uploaded_at.desc()) \
                       .first()
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
            fig = px.bar(df, x='category', y='amount', title='Спрос на товары')
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
    # получаем все файлы и отчёты
    files = File.query.filter_by(user_id=current_user.id).all()
    reports = (ForecastResult.query
               .options(joinedload(ForecastResult.file))
               .filter_by(user_id=current_user.id)
               .all())

    unique_cats = set()
    for f in files:
        if f.categories:
            unique_cats.update(f.categories)
    categories = sorted(unique_cats)

    return render_template('dashboard.html',
                           files=files,
                           reports=reports,
                           categories=categories)

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    try:
        # Получаем данные из запроса
        file_id = request.json.get('file_id')
        category = request.json.get('category')
        model_type = request.json.get('model')
        params = request.get_json()

        # Получаем файл из БД
        file = File.query.get(file_id)
        if not file:
            return jsonify({'error': 'File not found'}), 404
        
        # Читаем CSV
        df = pd.read_csv(BytesIO(file.data))
        df = df[df['category'] == category]
        
        if model_type == 'polynomial':
            # 1) Получаем degree и include_bias из запроса (дефолты: 3 и False)
            degree       = int(params.get('degree', 3))
            include_bias = bool(params.get('include_bias', False))
            future_days  = int(params.get('future_days', 30))
            # 2) Подготовка df
            df_prop = (
                df.rename(columns={'date': 'ds', 'value': 'y'})
                  .assign(ds=lambda d: pd.to_datetime(d['ds']))
            )

            # 3) Переводим дату в ordinal
            df_prop['ordinal'] = df_prop['ds'].map(datetime.datetime.toordinal)
            X = df_prop[['ordinal']].values
            y = df_prop['y'].values

            # 4) Создаём полиномиальные фичи с динамическими параметрами
            poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
            X_poly = poly.fit_transform(X)

            # 5) Обучаем линейную регрессию
            linreg = LinearRegression().fit(X_poly, y)

            # 6) Генерируем будущие даты (по умолчанию 30 дней, можно тоже сделать параметром)
            last_date    = df_prop['ds'].max()
            future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, future_days+1)]
            ordinals     = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
            X_future     = poly.transform(ordinals)

            # 7) Прогноз
            y_pred = linreg.predict(X_future)

            # 8) Формируем итоговый JSON
            forecast = pd.DataFrame({
                'ds':          future_dates,
                'yhat':        np.round(y_pred, 2),
                'yhat_lower':  np.round(y_pred, 2),
                'yhat_upper':  np.round(y_pred, 2),
            })

            result = {
                'historical': {
                    'dates':  df_prop['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': df_prop['y'].tolist()
                },
                'forecast': {
                    'dates':      forecast['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'yhat':       forecast['yhat'].tolist(),
                    'yhat_lower': forecast['yhat_lower'].tolist(),
                    'yhat_upper': forecast['yhat_upper'].tolist()
                }
            }
            insights = generate_insights(
                df=df_prop,                          # DataFrame с колонками ds и y
                forecast=forecast['yhat'].values,    # numpy‑массив ваших прогнозных точек
                future_dates=forecast['ds'],         # серии дат из forecast['ds']
                season_period=None                   # сезонность для polynomial не задаётся
            )
            result['insights'] = insights
            save_forecast_result(file_id, model_type, result)
            return jsonify(result)

        elif model_type == 'arima':
            # 1) Читаем из запроса
            m           = int(params.get('m',           7))   # длина сезона
            future_days = int(params.get('future_days', 30))  # горизонт прогноза

            # 2) Подготовка временного ряда
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').asfreq('D')
            df['value'].interpolate(method='time', inplace=True)
            ts = df['value']

            # 3) Автоподбор SARIMA с учётом m
            stepwise_model = auto_arima(
                ts,
                start_p=0, start_q=0,
                max_p=5,   max_q=5,
                d=None,    D=1,
                m=m,
                seasonal=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            # 4) Строим и тренируем модель
            order          = stepwise_model.order
            seasonal_order = stepwise_model.seasonal_order
            model          = ARIMA(ts, order=order, seasonal_order=seasonal_order)
            fitted         = model.fit()

            # 5) Делаем прогноз на future_days
            forecast_res = fitted.get_forecast(steps=future_days)
            mean_pred    = forecast_res.predicted_mean
            conf_int     = forecast_res.conf_int(alpha=0.05)

            # 6) Готовим даты прогноза
            last_date    = df.index.max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=future_days,
                freq='D'
            )

            # 7) Собираем результат
            result = {
                'historical': {
                    'dates': df.index.strftime('%Y-%m-%d').tolist(),
                    'values': ts.tolist()
                },
                'forecast': {
                    'dates':      future_dates.strftime('%Y-%m-%d').tolist(),
                    'yhat':       mean_pred.round(2).tolist(),
                    'yhat_lower': conf_int.iloc[:,0].round(2).tolist(),
                    'yhat_upper': conf_int.iloc[:,1].round(2).tolist()
                }
            }
            df_prop = pd.DataFrame({
                'ds': df.index,      # даты исторических точек
                'y':  ts.values      # сами значения ряда
            })

            # И вызываем его на основе mean_pred и будущих дат
            insights = generate_insights(
                df=df_prop,
                forecast=mean_pred.values,
                future_dates=future_dates,
                season_period=m
            )
            result['insights'] = insights
            save_forecast_result(file_id, model_type, result)
            return jsonify(result)

        elif model_type == 'lstm':
            LOOK_BACK      = int(params.get('look_back',     30))
            predict_range  = int(params.get('predict_range', 120))
            units1         = int(params.get('units1',        64))
            units2         = int(params.get('units2',        32))
            dropout        = float(params.get('dropout',      0.2))
            rec_dropout    = float(params.get('recurrent_dropout', 0.2))
            epoch_const    = int(params.get('epoch_const',   150))
            lr             = float(params.get('learning_rate', 0.001))

            # колбэки
            callbacks = [
                EarlyStopping(monitor='val_loss',
                            patience=int(epoch_const/3),
                            min_delta=1e-4,
                            restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss',
                                factor=0.5,
                                patience=10)
            ]

            # подготовка ряда…
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').asfreq('D')
            df['value'].interpolate(method='time', inplace=True)
            df = df.rename(columns={'value':'y'}).reset_index().rename(columns={'date':'ds'})

            # масштабирование
            scaler = MinMaxScaler((0,1))
            y_scaled = scaler.fit_transform(df['y'].values.reshape(-1,1))

            # создаём последовательности
            def create_sequences(data, look_back):
                X,y = [],[]
                for i in range(len(data)-look_back):
                    X.append(data[i:i+look_back,0])
                    y.append(data[i+look_back,0])
                return np.array(X), np.array(y)

            X, y_train = create_sequences(y_scaled, look_back=LOOK_BACK)
            X_train = X.reshape((X.shape[0], LOOK_BACK, 1))

            # строим модель с динамическими юнитами и dropout
            model = Sequential([
                LSTM(units1,
                    return_sequences=True,
                    dropout=dropout,
                    recurrent_dropout=rec_dropout,
                    input_shape=(LOOK_BACK,1)),
                LSTM(units2,
                    dropout=dropout,
                    recurrent_dropout=rec_dropout),
                Dense(1)
            ])
            optimizer = Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='mse')

            # обучение
            model.fit(
                X_train, y_train,
                epochs=epoch_const,
                batch_size=32,
                validation_split=0.1,
                shuffle=False,
                callbacks=callbacks
            )

            # прогноз
            last_sequence = y_scaled[-LOOK_BACK:].reshape(1,LOOK_BACK,1)
            forecast_scaled = []
            for _ in range(predict_range):
                pred = model.predict(last_sequence, verbose=0)
                forecast_scaled.append(pred[0,0])
                last_sequence = np.append(last_sequence[:,1:,:],
                                        pred.reshape(1,1,1),
                                        axis=1)

            forecast = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1,1)
            ).flatten()

            # даты и ответ
            last_date = df['ds'].max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                        periods=predict_range, freq='D')

            result = {
                'historical': {
                    'dates':  df['ds'].dt.strftime('%Y-%m-%d').tolist(),
                    'values': df['y'].tolist()
                },
                'forecast': {
                    'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                    'yhat':   [round(float(v),2) for v in forecast]
                }
            }
            # Правильный вызов generate_insights для LSTM:
            insights = generate_insights(
                df=df,
                forecast=forecast,
                future_dates=future_dates,
                season_period=None
            )
            result['insights'] = insights
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

def generate_insights(df, forecast, future_dates, season_period=None):
    insights = []

    # 1) Общий тренд прогноза
    if len(forecast) >= 2:
        first_fc, last_fc = forecast[0], forecast[-1]
        if first_fc != 0:
            change_pct = (last_fc - first_fc) / first_fc * 100
            if change_pct > 0:
                insights.append(
                    f"• Прогноз показывает рост продаж на {change_pct:.1f}% "
                    f"в период с {future_dates[0].strftime('%d-%m-%Y')} по {future_dates[-1].strftime('%%d-%m-%Y')}."
                )
            else:
                insights.append(
                    f"• Ожидается снижение продаж на {abs(change_pct):.1f}% — "
                    "рассмотрите акции или спецпредложения для поддержки уровня продаж."
                )

    # 2) Топ‑3 и низ‑3 дня
    import numpy as np
    idx = np.argsort(forecast)
    best3  = sorted(idx[-3:]) if len(forecast) >= 3 else idx
    worst3 = sorted(idx[:3])  if len(forecast) >= 3 else idx
    best_days  = ", ".join(future_dates[i].strftime('%d-%m-%Y') for i in best3)
    worst_days = ", ".join(future_dates[i].strftime('%d-%m-%Y') for i in worst3)
    if best_days:
        insights.append(
            f"• Пики продаж ожидаются: {best_days}. "
            "Запланируйте маркетинговые активности именно в эти дни."
        )
    if worst_days:
        insights.append(
            f"• Наименьшие продажи: {worst_days}. "
            "Для этих дат рекомендуется подготовить особые предложения."
        )

    # 3) Аномалии в исторических данных (опционально)
    if season_period:
        from statsmodels.tsa.seasonal import seasonal_decompose
        try:
            dec   = seasonal_decompose(df.set_index('ds')['y'], model='additive', period=season_period)
            resid = dec.resid.dropna()
            std   = resid.std()
            anom  = resid[abs(resid) > 2 * std].index.strftime('%Y-%m-%d').tolist()
            if anom:
                insights.append(
                    "• Обнаружены аномалии в исторических данных: " +
                    ", ".join(anom[:5]) +
                    (", …" if len(anom) > 5 else "") +
                    ". Проверьте события в эти даты."
                )
        except Exception:
            pass

    # 4) Совет по сезонности
    if season_period:
        insights.append(
            f"• Учитывайте сезонность с периодом {season_period} — "
            "планируйте кампании заранее перед ожидаемыми пиками."
        )

    return insights

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

@app.route('/recommendations', methods=['GET'])
@login_required
def recommendations():
    # получаем report_id из query‑string
    report_id = request.args.get('report_id')
    if not report_id:
        return jsonify({'error': 'report_id не задан'}), 400

    rpt = ForecastResult.query.get(report_id)
    if not rpt:
        return jsonify({'error': 'Отчёт не найден'}), 200

    insights = rpt.report_data.get('insights') or []
    return jsonify({'insights': insights})

if __name__ == '__main__':
    app.run(debug=True)