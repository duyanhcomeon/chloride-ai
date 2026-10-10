"""Transparent supervised regression for measured D28. No pretrained scientific model."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ['fly_ash_pct', 'water_binder_ratio', 'curing_days', 'compressive_strength_mpa', 'temperature_c']
TARGET = 'd28_m2_s'

def fit_model(df):
    required = FEATURES + [TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(missing))
    clean = df[required].apply(pd.to_numeric, errors='coerce').dropna()
    clean = clean.loc[(clean[TARGET]>0) & (clean['water_binder_ratio']>0) & (clean['curing_days']>0)]
    if len(clean) < 30:
        raise ValueError('Need at least 30 valid samples to run this demonstration')
    x_train, x_test, y_train, y_test = train_test_split(clean[FEATURES], np.log10(clean[TARGET]), test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(x_train,y_train)
    pred = 10**model.predict(x_test)
    observed = 10**y_test.to_numpy()
    metrics = {'n':len(clean), 'MAE_m2_s':float(mean_absolute_error(observed,pred)), 'R2_log10':float(r2_score(y_test, model.predict(x_test)))}
    bounds = {f:(float(clean[f].min()),float(clean[f].max())) for f in FEATURES}
    return model, metrics, bounds

def predict(model, values, bounds=None):
    row = pd.DataFrame([{key:float(values[key]) for key in FEATURES}])
    out_of_range = [f for f in FEATURES if bounds and not(bounds[f][0] <= float(values[f]) <= bounds[f][1])]
    return float(10**model.predict(row)[0]), out_of_range
