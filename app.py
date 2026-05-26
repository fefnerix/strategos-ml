import os
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
SECRET = os.getenv('STRATEGOS_SECRET', '')

def require_secret(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if SECRET and request.headers.get('X-Strategos-Secret','') != SECRET:
            return jsonify({'error':'unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/health')
def health():
    return jsonify({'status':'ok','service':'strategos-ml'})

@app.route('/forecast', methods=['POST'])
@require_secret
def forecast():
    body = request.get_json()
    data = body.get('data', [])
    if len(data) < 14:
        return jsonify({'error': 'min 14 pontos', 'received': len(data)}), 422
    try:
        from prophet import Prophet
        import pandas as pd
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'], errors='coerce').fillna(0)
        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )
        m.fit(df)
        horizon = int(body.get('horizon_days', 30))
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future)
        result = fc.tail(horizon)[['ds','yhat','yhat_lower','yhat_upper']].copy()
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
        return jsonify({
            'funnel_id': body.get('funnel_id'),
            'metric': body.get('metric', 'roas'),
            'model': 'prophet_v1',
            'forecast': result.round(4).to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
