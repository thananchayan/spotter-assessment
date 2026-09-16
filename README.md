# Freight rate prediction

Train and evaluate a freight-rate model, then predict the 12,000 supplied November-December 2025 loads.

## Setup

Use Python 3.12. From the repository root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux, use `.venv/bin/python` instead of `.\.venv\Scripts\python.exe`.

## Run and test

```powershell
.\.venv\Scripts\python.exe solution.py
.\.venv\Scripts\python.exe score.py --predictions output/validation_predictions.csv --december-predictions output/december_predictions.csv --output-dir output/scorer_results
.\.venv\Scripts\python.exe verify_outputs.py
```

The first command trains the models and writes outputs. The scorer validates 12,000 final predictions and 31 December predictions, then generates the chart. The last command checks ID order, unchanged scenario inputs, positive rates, invalid-weight handling and split accounting. Training takes several minutes. The scorer does not measure hidden final accuracy.

## Inputs

| File | Purpose |
|---|---|
| `train-test.csv` | 48,000 labeled January-October loads; target is total `posted_rate` in dollars |
| `validation.csv` | 12,000 unlabeled November-December loads |
| `validation-predictions-template.csv` | Required output IDs and order |
| `december-chart-inputs.csv` | Fixed Lexington to Fort Wayne scenario for every December date |

Inputs are preserved. The filenames above match the supplied files.

## Method

- Non-positive weights become missing. Numeric imputation uses training-fold medians. Categorical encoding tolerates unseen values.
- Features include distance, equipment, cleaned weight, geography and calendar patterns. Signal variants add market index and quote signal. IDs are excluded.
- Models predict log rate per mile, then convert predictions back to total dollars. Extreme positive labels remain in training and evaluation.
- Candidates train on January-June, with model selection on July-August MAE. The selected model is refitted through August and evaluated on September-October without retuning. Final inference uses a refit on all labeled rows.
- The December scenario lacks market and quote signals. Its separate core model is selected on the same July-August window. Fixed coordinates are looked up from development data.

## Results

| Model | July-August MAE | September-October MAE |
|---|---:|---:|
| Median rate per mile | $249.01 | $256.95 |
| Quote times distance | $367.29 | $246.02 |
| Ridge with core inputs | $397.58 | Not evaluated |
| Boosting with core inputs | $207.93 | $113.76 |
| Boosting with market index | $190.43 | Not evaluated |
| Boosting with market and quote, selected | $171.46 | $127.91 |

Selected-model test MAPE is 5.47%, RMSE is $638.90 and R-squared is 0.8247. The core model did better on the later test; the original selection is retained to avoid selecting on test results. These are internal temporal test results, not final hidden validation metrics.

Quote reliability changes by month, future loads introduce new cities, and large labels inflate RMSE. Confirm signal availability before pricing and evaluate on fresh future labels before production use. Less than one annual cycle limits conclusions about seasonality.

## Outputs and code

- `output/validation_predictions.csv`: submission file with exactly `load_id,predicted_rate`.
- `output/december_predictions.csv`: fixed scenario predictions.
- `output/scorer_results/candidate_december.png`: chart from the unmodified scorer.
- `output/metrics.csv` and `output/model_metadata.json`: evaluation results and configuration.
- `solution.py`: feature engineering, model pipelines, temporal validation and prediction.
- `verify_outputs.py`: contract and data-cleaning checks.

Training also creates local audit tables, held-out predictions and `model.joblib`. These are reproducible and excluded from version control. Only load trusted model files.
