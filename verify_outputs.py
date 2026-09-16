"""Meaningful submission contract and model inference checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from score import validate_predictions, validate_december
from solution import features

root=Path(__file__).resolve().parent
out=root/'output'
p=pd.read_csv(out/'validation_predictions.csv')
t=pd.read_csv(root/'validation-predictions-template.csv')
d=pd.read_csv(out/'december_predictions.csv')
original=pd.read_csv(root/'december-chart-inputs.csv')
validate_predictions(p)
validate_december(d)
assert p.load_id.equals(t.load_id), 'Template order changed'
pd.testing.assert_frame_equal(d.drop(columns='predicted_rate'),original.drop(columns='predicted_rate'))
# Invalid and missing weights must have identical model inputs, not abs(weight).
probe=pd.DataFrame({'date':['2025-12-01','2025-12-01'],'distance':[360,360],'weight':[-100,np.nan]})
f=features(probe)
assert f.weight.isna().all() and f.weight_missing.eq(1).all()
assert 'load_id' not in f and 'posted_rate' not in f
meta=json.loads((out/'model_metadata.json').read_text())
assert sum(meta['split_rows'][k] for k in ['Jan_Jun_train','Jul_Aug_selection','Sep_Oct_test'])==48000
assert (out/'scorer_results/candidate_december.png').is_file()
print('PASS: 12,000 IDs/order/positive rates; 31 unchanged scenarios; invalid-weight handling; split accounting; chart.')
