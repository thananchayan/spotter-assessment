"""Reproducible chronological freight-rate assessment. Run: python solution.py."""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
CATS = ['pickup', 'delivery', 'equipment']


def features(frame, variant='core'):
    """Deterministic row features; no labels, IDs, or fitted global statistics."""
    x = frame.reindex(columns=CATS + ['pickup_lat', 'pickup_lon', 'delivery_lat',
        'delivery_lon', 'distance', 'weight']).copy()
    x['weight'] = x.weight.where(x.weight > 0)
    x['weight_missing'] = x.weight.isna().astype(int)
    dt = pd.to_datetime(frame.date, errors='raise')
    x['weekday'] = dt.dt.dayofweek.astype(str)
    x['day_of_month'] = dt.dt.day
    x['time'] = (dt - pd.Timestamp('2025-01-01')).dt.days
    for k in [1, 2]:
        x[f'season_sin_{k}'] = np.sin(2*np.pi*k*dt.dt.dayofyear/365.25)
        x[f'season_cos_{k}'] = np.cos(2*np.pi*k*dt.dt.dayofyear/365.25)
    x['log_distance'] = np.log(x.distance)
    x['inverse_distance'] = 1/x.distance
    x['delta_lat'] = x.delivery_lat-x.pickup_lat
    x['delta_lon'] = x.delivery_lon-x.pickup_lon
    if variant in ['market', 'quote']:
        x['market_index'] = frame.reindex(columns=['market_index']).market_index
    if variant == 'quote':
        x['quote_signal'] = frame.reindex(columns=['quote_signal']).quote_signal
    return x


def make_model(name):
    variant = name.split('_')[-1]
    cats = CATS + ['weekday']
    columns = features(pd.DataFrame({'date':['2025-01-01'],'distance':[360],
        'weight':[32000]}), variant).columns
    nums = [c for c in columns if c not in cats]
    numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler())
    pre = ColumnTransformer([
        ('num', numeric, nums),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
    if name.startswith('ridge'):
        reg = Ridge(alpha=30)
    else:
        reg = HistGradientBoostingRegressor(loss='squared_error', max_iter=250,
            learning_rate=0.07, max_leaf_nodes=23, min_samples_leaf=35,
            l2_regularization=8, early_stopping=False, random_state=42)
    return make_pipeline(pre, reg)


def fit_model(name, frame):
    model = make_model(name)
    # Log rate per mile reduces domination by a few extreme total-price labels.
    model.fit(features(frame, name.split('_')[-1]), np.log(frame.posted_rate/frame.distance))
    return model


def predict(model, name, frame):
    return np.maximum(0.01, np.exp(model.predict(features(frame, name.split('_')[-1]))) * frame.distance)


def metrics(y, pred):
    return {'MAE':float(mean_absolute_error(y,pred)),
        'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
        'MAPE_pct':float(np.mean(np.abs((y-pred)/y))*100),
        'R2':float(r2_score(y,pred))}


def audit(frame):
    result = {'rows':len(frame),'columns':list(frame.columns),
        'missing':frame.isna().sum().astype(int).to_dict(),
        'duplicate_rows':int(frame.duplicated().sum()),
        'duplicate_ids':int(frame.load_id.duplicated().sum()) if 'load_id' in frame else None}
    if 'date' in frame:
        result.update(date_min=frame.date.min(), date_max=frame.date.max(),
            invalid_weight=int((frame.weight<=0).sum()),
            nonpositive_distance=int((frame.distance<=0).sum()))
    return result


def enrich_december(december, train):
    """Use training-only city lookup. Dynamic signals absent: core model required."""
    cities = pd.concat([
        train[['pickup','pickup_lat','pickup_lon']].set_axis(['city','lat','lon'],axis=1),
        train[['delivery','delivery_lat','delivery_lon']].set_axis(['city','lat','lon'],axis=1)])
    lookup = cities.groupby('city')[['lat','lon']].median()
    result = december.copy()
    for role in ['pickup','delivery']:
        for coord in ['lat','lon']:
            result[f'{role}_{coord}'] = result[role].map(lookup[coord])
    assert result.filter(regex='_lat|_lon').notna().all().all()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=ROOT)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'output')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    train = pd.read_csv(args.data_dir/'train-test.csv')
    final = pd.read_csv(args.data_dir/'validation.csv')
    template = pd.read_csv(args.data_dir/'validation-predictions-template.csv')
    december = pd.read_csv(args.data_dir/'december-chart-inputs.csv')
    assert len(train)==48000 and len(final)==12000
    assert train.load_id.is_unique and final.load_id.is_unique and template.load_id.is_unique
    assert set(final.load_id)==set(template.load_id)
    assert set(train.load_id).isdisjoint(final.load_id)
    assert (train.posted_rate>0).all() and (train.distance>0).all() and (final.distance>0).all()
    assert train.date.max()<final.date.min()
    quality = {n:audit(f) for n,f in [('train-test',train),('validation',final),
        ('template',template),('december',december)]}
    for c in CATS:
        quality['validation'][f'unseen_{c}'] = sorted(set(final[c])-set(train[c]))
    train_lanes = set(zip(train.pickup,train.delivery))
    quality['validation']['unseen_lane_rows'] = sum((a,b) not in train_lanes for a,b in zip(final.pickup,final.delivery))
    quality['train-test']['rate_summary'] = train.posted_rate.describe(percentiles=[.01,.5,.95,.99]).to_dict()
    quality['train-test']['numeric_summary'] = train.select_dtypes('number').describe().to_dict()
    quality['validation']['numeric_summary'] = final.select_dtypes('number').describe().to_dict()
    monthly = train.assign(month=train.date.str[:7],rpm=train.posted_rate/train.distance,
        quote_error=abs(train.posted_rate-train.quote_signal*train.distance)).groupby('month').agg(
        rows=('load_id','size'),median_rate=('posted_rate','median'),median_rpm=('rpm','median'),
        quote_MAE=('quote_error','mean'),median_market=('market_index','median'))
    monthly.to_csv(out/'monthly_analysis.csv')
    train.groupby('equipment').posted_rate.agg(['count','mean','median']).to_csv(out/'equipment_analysis.csv')
    (out/'data_audit.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')

    # Select on July-August only; September-October is a later locked test.
    fit = train[train.date<'2025-07-01']
    tune = train[(train.date>='2025-07-01')&(train.date<'2025-09-01')]
    pretest = train[train.date<'2025-09-01']
    test = train[train.date>='2025-09-01']
    names = ['ridge_core','hist_core','hist_market','hist_quote']
    rows = []
    baseline_rpm = (fit.posted_rate/fit.distance).median()
    rows.append({'model':'median_rpm','split':'selection_Jul_Aug',**metrics(tune.posted_rate,tune.distance*baseline_rpm)})
    rows.append({'model':'quote_times_distance','split':'selection_Jul_Aug',**metrics(tune.posted_rate,tune.distance*tune.quote_signal)})
    for name in names:
        print('Training selection candidate:',name,flush=True)
        model = fit_model(name,fit)
        rows.append({'model':name,'split':'selection_Jul_Aug',**metrics(tune.posted_rate,predict(model,name,tune))})
        print(rows[-1],flush=True)
    scores = {r['model']:r['MAE'] for r in rows if r['model'] in names}
    selected = min(scores,key=scores.get)
    # Chart must use available inputs. Choose core independently on same selection period.
    core = min(['ridge_core','hist_core'],key=scores.get)
    print('Locked final model:',selected,'December model:',core,flush=True)
    heldout = test[['load_id','date','equipment','pickup','delivery','distance','posted_rate']].copy()
    for name in dict.fromkeys([selected,core]):
        model = fit_model(name,pretest)
        pred = predict(model,name,test)
        rows.append({'model':name,'split':'locked_test_Sep_Oct',**metrics(test.posted_rate,pred)})
        heldout[f'predicted_{name}'] = pred
    for name,pred in [('median_rpm',test.distance*(pretest.posted_rate/pretest.distance).median()),
                      ('quote_times_distance',test.distance*test.quote_signal)]:
        rows.append({'model':name,'split':'locked_test_Sep_Oct',**metrics(test.posted_rate,pred)})
    results = pd.DataFrame(rows)
    results.to_csv(out/'metrics.csv',index=False)
    heldout.to_csv(out/'heldout_predictions.csv',index=False)
    groups = []
    test_pred = heldout[f'predicted_{selected}']
    pretest_lanes = set(zip(pretest.pickup,pretest.delivery))
    for dimension, labels in [('month',test.date.str[:7]),('equipment',test.equipment),
        ('lane_seen',pd.Series(['seen' if (a,b) in pretest_lanes else 'unseen'
            for a,b in zip(test.pickup,test.delivery)],index=test.index))]:
        for value, idx in labels.groupby(labels).groups.items():
            groups.append({'dimension':dimension,'group':value,'rows':len(idx),**metrics(test.loc[idx,'posted_rate'],test_pred.loc[idx])})
    pd.DataFrame(groups).to_csv(out/'test_segments.csv',index=False)
    models = {}
    for name in dict.fromkeys([selected,core]):
        print('Refitting all labeled data:',name,flush=True)
        models[name] = fit_model(name,train)
    mapping = pd.Series(np.asarray(predict(models[selected],selected,final)),index=final.load_id)
    template['predicted_rate'] = template.load_id.map(mapping).round(2)
    template.to_csv(out/'validation_predictions.csv',index=False)
    december['predicted_rate'] = np.asarray(predict(models[core],core,enrich_december(december,train))).round(2)
    december.to_csv(out/'december_predictions.csv',index=False)
    joblib.dump({'models':models,'selected':selected,'december_model':core},out/'model.joblib')
    metadata = {'selected_model':selected,'december_model':core,'seed':42,
        'selection_rule':'Lowest July-August MAE among four trained candidates; core-only chart selected separately.',
        'split_rows':{'Jan_Jun_train':len(fit),'Jul_Aug_selection':len(tune),
            'Jan_Aug_refit':len(pretest),'Sep_Oct_test':len(test),'final_refit':len(train)},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
            'pandas':pd.__version__,'scikit-learn':sklearn.__version__},
        'december_range':[float(december.predicted_rate.min()),float(december.predicted_rate.max())]}
    (out/'model_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(results.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
