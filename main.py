import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from flask import Flask, render_template, request, send_file
import os
from datetime import datetime, timedelta
app = Flask(__name__, static_folder='.')

data = pd.read_csv("BTCUSDT_converted.csv", sep=";")
data.columns = ['timestamp', 'symbol', 'price']
data['timestamp'] = pd.to_datetime(data['timestamp'], format='%d.%m.%Y:%H.%M.%S')
data['price'] = data['price'].str.replace(',', '.').astype(float)

data['price_change'] = data['price'].diff()
data['ma_5'] = data['price'].rolling(window=5).mean()

last_date = data['timestamp'].max()
test_start = last_date - timedelta(days=2)
train_end = test_start - timedelta(seconds=1)

train_data = data[data['timestamp'] <= train_end]
test_data = data[data['timestamp'] >= test_start].iloc[::60]  # Интервал 1 минута

data['target_60s'] = data['price'].shift(-60)
data['target_3600s'] = data['price'].shift(-3600)

features = ['price', 'price_change', 'ma_5']
X_train = train_data[features].dropna()
X_test = test_data[features].dropna()

y_train_60s = data.loc[X_train.index, 'target_60s']
y_train_3600s = data.loc[X_train.index, 'target_3600s']
y_test_60s = data.loc[X_test.index, 'target_60s']
y_test_3600s = data.loc[X_test.index, 'target_3600s']

@app.route('/', methods=['GET', 'POST'])
def index():
    graph_60s_html = None
    graph_3600s_html = None
    if request.method == 'POST':
        n_estimators = int(request.form['n_estimators'])
        max_depth = int(request.form['max_depth'])
        n_jobs = int(request.form['n_jobs']) if 'n_jobs' in request.form else 3  # Значение по умолчанию 3

        # Обучение
        model_60s = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=n_jobs)
        model_3600s = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=n_jobs)
        model_60s.fit(X_train, y_train_60s)
        model_3600s.fit(X_train, y_train_3600s)

        # Прогноз
        predictions_60s = model_60s.predict(X_test)
        predictions_3600s = model_3600s.predict(X_test)

        # Метрики
        mask_60s = ~y_test_60s.isna() & ~pd.isna(predictions_60s)
        mask_3600s = ~y_test_3600s.isna() & ~pd.isna(predictions_3600s)
        mae_60s = mean_absolute_error(y_test_60s[mask_60s], predictions_60s[mask_60s])
        mse_60s = mean_squared_error(y_test_60s[mask_60s], predictions_60s[mask_60s])
        r2_60s = r2_score(y_test_60s[mask_60s], predictions_60s[mask_60s])
        mae_3600s = mean_absolute_error(y_test_3600s[mask_3600s], predictions_3600s[mask_3600s])
        mse_3600s = mean_squared_error(y_test_3600s[mask_3600s], predictions_3600s[mask_3600s])
        r2_3600s = r2_score(y_test_3600s[mask_3600s], predictions_3600s[mask_3600s])

        # Результаты
        results_60s = pd.DataFrame({
            'timestamp': test_data.loc[X_test.index, 'timestamp'],
            'price': test_data.loc[X_test.index, 'price'],
            'predicted_60s': predictions_60s,
            'actual_60s': y_test_60s
        })
        results_3600s = pd.DataFrame({
            'timestamp': test_data.loc[X_test.index, 'timestamp'],
            'price': test_data.loc[X_test.index, 'price'],
            'predicted_3600s': predictions_3600s,
            'actual_3600s': y_test_3600s
        })
        results_60s.to_excel('predictions_60s_minute_interval.xlsx', index=False)
        results_3600s.to_excel('predictions_3600s_hour_interval.xlsx', index=False)

        # Графики
        fig_60s = go.Figure()
        fig_60s.add_trace(go.Scatter(x=results_60s['timestamp'], y=results_60s['predicted_60s'], name='Predicted 60s', line=dict(color='#1E90FF')))
        fig_60s.add_trace(go.Scatter(x=results_60s['timestamp'], y=results_60s['actual_60s'], name='Actual 60s', line=dict(color='#FF4500')))
        fig_60s.update_layout(
            title=f'Prediction vs Actual for 60s (MAE: {mae_60s:.2f}, MSE: {mse_60s:.2f}, R²: {r2_60s:.2f})',
            xaxis_title='Timestamp',
            yaxis_title='Price (USDT)',
            legend_title_text='Series',
            template='plotly_dark'
        )

        fig_3600s = go.Figure()
        fig_3600s.add_trace(go.Scatter(x=results_3600s['timestamp'], y=results_3600s['predicted_3600s'], name='Predicted 3600s', line=dict(color='#1E90FF')))
        fig_3600s.add_trace(go.Scatter(x=results_3600s['timestamp'], y=results_3600s['actual_3600s'], name='Actual 3600s', line=dict(color='#FF4500')))
        fig_3600s.update_layout(
            title=f'Prediction vs Actual for 3600s (MAE: {mae_3600s:.2f}, MSE: {mse_3600s:.2f}, R²: {r2_3600s:.2f})',
            xaxis_title='Timestamp',
            yaxis_title='Price (USDT)',
            legend_title_text='Series',
            template='plotly_dark'
        )

        graph_60s_html = fig_60s.to_html(include_plotlyjs='cdn', full_html=False)
        graph_3600s_html = fig_3600s.to_html(include_plotlyjs='cdn', full_html=False)

        return render_template('index.html', graph_60s=graph_60s_html, graph_3600s=graph_3600s_html,
                              log_60s='predictions_60s_minute_interval.xlsx',
                              log_3600s='predictions_3600s_hour_interval.xlsx')

    return render_template('index.html')

@app.route('/logs/<filename>')
def display_log(filename):
    if filename in ['predictions_60s_minute_interval.xlsx', 'predictions_3600s_hour_interval.xlsx']:
        df = pd.read_excel(filename)
        html_table = df.to_html(classes='table table-striped', index=False)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ background-color: #000; color: #fff; font-family: Arial, sans-serif; }}
                table {{ width: 80%; margin: 20px auto; border-collapse: collapse; }}
                th, td {{ border: 1px solid #fff; padding: 8px; text-align: left; }}
                th {{ background-color: #1E90FF; }}
            </style>
        </head>
        <body>
            <h1 style="text-align: center;">{filename.replace('.xlsx', '').replace('_', ' ').title()}</h1>
            {html_table}
        </body>
        </html>
        """
    return "File not found", 404

@app.route('/download/<filename>')
def download_log(filename):
    if filename in ['predictions_60s_minute_interval.xlsx', 'predictions_3600s_hour_interval.xlsx']:
        return send_file(filename, as_attachment=True)
    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)