import os
from functools import wraps
from flask import Flask, request, jsonify

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
    return jsonify({'status':'ok','version':'2.0.0',
      'endpoints':['/forecast','/kalman','/stl','/changepoint','/bandit','/nlp/copy']})

# ── PILAR 2: PROPHET FORECAST ────────────────────────────────────────
@app.route('/forecast', methods=['POST'])
@require_secret
def forecast():
    body = request.get_json()
    data = body.get('data',[])
    if len(data) < 14:
        return jsonify({'error':'min 14 pontos','received':len(data)}), 422
    try:
        from prophet import Prophet
        import pandas as pd
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        df['y'] = pd.to_numeric(df['y'],errors='coerce').fillna(0)
        m = Prophet(yearly_seasonality=False,weekly_seasonality=True,
                    changepoint_prior_scale=0.05,interval_width=0.95)
        m.fit(df)
        horizon = int(body.get('horizon_days',30))
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future)
        result = fc.tail(horizon)[['ds','yhat','yhat_lower','yhat_upper']].copy()
        result['ds'] = result['ds'].dt.strftime('%Y-%m-%d')
        return jsonify({'funnel_id':body.get('funnel_id'),
          'metric':body.get('metric','roas'),'model':'prophet_v1',
          'forecast':result.round(4).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ── PILAR 2: KALMAN FILTER ───────────────────────────────────────────
@app.route('/kalman', methods=['POST'])
@require_secret
def kalman():
    import numpy as np
    body = request.get_json()
    series = body.get('data',[])
    if not series:
        return jsonify({'error':'data required'}), 400
    Q, R = 1e-5, 0.01
    values = [float(p['roas']) for p in series]
    x, P, out = values[0], 1.0, []
    for z in values:
        P += Q; K = P/(P+R); x += K*(z-x); P *= (1-K)
        out.append(round(float(x),4))
    return jsonify({'funnel_id':body.get('funnel_id'),
      'smoothed':[{'date':series[i]['date'],'roas_kalman':v}
                  for i,v in enumerate(out)]})

# ── PILAR 2: STL DECOMPOSITION ───────────────────────────────────────
@app.route('/stl', methods=['POST'])
@require_secret
def stl():
    body = request.get_json()
    data = body.get('data',[])
    if len(data) < 14:
        return jsonify({'error':'min 14 pontos'}), 422
    try:
        import pandas as pd, numpy as np
        from statsmodels.tsa.seasonal import STL as STLModel
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        r = STLModel(df['roas'],period=7,robust=True).fit()
        std = float(np.std(r.resid))
        return jsonify({'funnel_id':body.get('funnel_id'),
          'decomposition':[
            {'date':str(d.date()),'trend':round(float(t),4),
             'seasonal':round(float(s),4),'residual':round(float(res),4),
             'is_anomaly':abs(float(res))>2*std}
            for d,t,s,res in zip(r.trend.index,r.trend.values,
                                  r.seasonal.values,r.resid.values)]})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

# ── PILAR 1: CHANGEPOINT (CUSUM) ─────────────────────────────────────
@app.route('/changepoint', methods=['POST'])
@require_secret
def changepoint():
    import numpy as np
    body = request.get_json()
    data = body.get('data',[])
    if len(data) < 21:
        return jsonify({'error':'min 21 pontos'}), 422
    values = [float(p['value']) for p in data]
    dates  = [p['date'] for p in data]
    mu = float(np.mean(values[:14]))
    sigma = max(float(np.std(values[:14])),0.01)
    cps, cp, cn = [], 0.0, 0.0
    for i in range(14,len(values)):
        z = (values[i]-mu)/sigma
        cp = max(0,cp+z-0.5); cn = max(0,cn-z-0.5)
        if cp > 4.0:
            cps.append({'date':dates[i],'change_type':'upward',
              'magnitude':round(float(values[i]-mu),4),
              'confidence':round(min(cp/4.0,1.0),4)})
            cp=0.0; mu=float(np.mean(values[max(0,i-7):i+1]))
        if cn > 4.0:
            cps.append({'date':dates[i],'change_type':'downward',
              'magnitude':round(float(values[i]-mu),4),
              'confidence':round(min(cn/4.0,1.0),4)})
            cn=0.0; mu=float(np.mean(values[max(0,i-7):i+1]))
    return jsonify({'funnel_id':body.get('funnel_id'),
      'metric':body.get('metric','roas'),'changepoints':cps})

# ── PILAR 3: MULTI-ARMED BANDIT (Thompson Sampling) ──────────────────
@app.route('/bandit', methods=['POST'])
@require_secret
def bandit():
    import numpy as np
    body = request.get_json()
    creatives = body.get('creatives',[])
    if not creatives:
        return jsonify({'error':'creatives required'}), 400
    results = []
    for c in creatives:
        conv = max(int(c.get('conversions',0)),0)
        imp  = max(int(c.get('impressions',1)),1)
        freq = float(c.get('frequency',1.0))
        ctr  = float(c.get('ctr',1.0))
        a,b  = conv+1,(imp-conv)+1
        exp  = a/(a+b)
        # Thompson Sampling real com distribuição Beta
        ub   = float(np.percentile(np.random.beta(a,b,10000),95))
        fatigue = max(0,(freq-3.0)*0.05)
        status = ('retire' if (freq>5 or ctr<0.5) else
                  'exploit' if exp>0.03 else 'explore')
        results.append({'ad_id':c['ad_id'],'ad_name':c.get('ad_name'),
          'alpha':a,'beta':b,'expected_value':round(exp,6),
          'upper_bound':round(ub-fatigue,6),'status':status})
    results.sort(key=lambda x:x['upper_bound'],reverse=True)
    for i,r in enumerate(results): r['bandit_rank']=i+1
    return jsonify({'funnel_id':body.get('funnel_id'),'rankings':results})

# ── PILAR 3: NLP COPY ANALYSIS ───────────────────────────────────────
@app.route('/nlp/copy', methods=['POST'])
@require_secret
def nlp_copy():
    import re
    body = request.get_json()
    creatives = body.get('creatives',[])
    if not creatives:
        return jsonify({'error':'creatives required'}), 400
    cpas = [float(c.get('cpa_brl',0)) for c in creatives if c.get('cpa_brl')]
    med  = sorted(cpas)[len(cpas)//2] if cpas else 100
    results = []
    for c in creatives:
        txt = (c.get('primary_text') or '')+' '+(c.get('headline') or '')
        t   = txt.lower()
        cpa = float(c.get('cpa_brl',med))
        tier = ('top' if cpa<med*0.8 else 'low' if cpa>med*1.3 else 'mid')
        hook = ('pergunta' if re.search(r'\?',txt[:80]) else
                'numero'   if re.search(r'\b\d+\b',txt[:60]) else
                'afirmacao_choque' if any(w in t for w in ['erro','problema','culpa','falha']) else
                'empatia'  if any(w in t for w in ['entendo','sei como','ja estive']) else
                'autoridade')
        cta  = ('urgencia'    if any(w in t for w in ['ultimas vagas','so hoje','encerra','restam']) else
                'beneficio'   if any(w in t for w in ['resultado','conquiste','alcance']) else
                'social_proof'if any(w in t for w in ['alunos','pessoas','aprovados']) else
                'curiosidade')
        emotion = ('medo'        if any(w in t for w in ['medo','risco','perde','falhou']) else
                   'aspiracao'   if any(w in t for w in ['sonho','conquista','sucesso','liberdade']) else
                   'pertencimento' if any(w in t for w in ['comunidade','junto','familia']) else
                   'ganancia')
        results.append({'ad_id':c['ad_id'],'hook_type':hook,'cta_type':cta,
          'emotion_tone':emotion,
          'has_number':bool(re.search(r'\b\d+\b',txt)),
          'has_social_proof':any(w in t for w in ['alunos','pessoas','aprovados']),
          'has_urgency':any(w in t for w in ['ultimas','so hoje','encerra']),
          'word_count':len(txt.split()),'performance_tier':tier})
    return jsonify({'funnel_id':body.get('funnel_id'),'insights':results})

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=int(os.getenv('PORT',5000)))
