# AMZN Forecast Model Comparison

## Results by horizon

| Horizon | Linear MAE | Random Forest MAE | Linear RMSE | Random Forest RMSE | Linear Direction | Random Forest Direction | Folds |
|:--|--:|--:|--:|--:|--:|--:|--:|
| n+0 | 17.41 | 13.71 | 18.51 | 14.74 | 0.0% | 0.0% | 83 / 83 |
| n+1 | 17.96 | 13.89 | 19.24 | 14.98 | 51.1% | 48.9% | 83 / 83 |
| n+2 | 18.58 | 14.13 | 19.93 | 15.29 | 50.7% | 47.8% | 82 / 82 |
| n+3 | 19.15 | 14.47 | 20.51 | 15.68 | 49.0% | 46.3% | 82 / 82 |
| n+4 | 19.48 | 14.92 | 20.77 | 16.27 | 48.5% | 47.6% | 82 / 82 |
| n+5 | 19.67 | 15.63 | 20.90 | 16.85 | 50.5% | 50.0% | 82 / 82 |

Folds are shown as linear / Random Forest. MAE and RMSE are in the ticker's price units; lower is better. Direction accuracy measures whether the predicted change has the same sign as the actual change.

## Summary

Across n+1 to n+5, Random Forest had the lower mean MAE over 5 paired horizons (14.61 vs. 18.97); it had lower MAE in 5 of those horizons. Mean RMSE was 15.81 for Random Forest and 20.27 for linear regression. Mean direction accuracy was 48.1% and 50.0%, respectively.

## Pipeline data coverage and impact
The pipeline found 61 distinct dates with both tweets and price bars. Features, forecast comparison, sentiment summary, and chart were generated from the final imported data.

n+0 is the current-period close, not a future horizon, so its direction score is not evidence of future directional skill. Judge future performance using n+1 through n+5 and chronological out-of-sample results.

### Interpretation

The report compares linear regression with a Random Forest using only engineered sentiment and price features. Each model and horizon uses the same rolling-origin folds, keeping validation chronological and preventing future information from entering training data.

The Random Forest uses 100 trees, a minimum leaf size of four, and a fixed seed to limit variance and make results reproducible. It can learn nonlinear relationships, but added complexity does not guarantee better forecasts. Direction accuracy near 50% is approximately chance-level and should not be treated as trading evidence without substantially more out-of-sample data.
