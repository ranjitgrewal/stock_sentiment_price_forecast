# AMZN Trading Risk Evaluation

## Methodology

Risk metrics are calculated from chronological out-of-sample predictions generated with the same feature columns, model fitting functions, rolling training window, and five-row validation batches used by the forecast model.

For each forecast horizon n+h (h=1..5), the strategy takes a long position when predicted future close is above the current close, a short position when it is below, and no position when equal. Realized trade return is the signed return from current close to the actual n+h close. To avoid overlapping holding periods, the report uses every h-th out-of-sample forecast. Sharpe and Sortino are annualized using 252/h periods per year. Entry and exit costs are deducted using the configured one-way transaction cost.

- Annual risk-free rate: 2.00%
- One-way transaction cost: 0.100%
- Training window: 60 rows
- Validation window: 5 rows
- Horizons included: n+1 through n+5 (n+0 excluded)

## Risk and strategy performance

| Model | Horizon | Sharpe Ratio | Sortino Ratio | Max Drawdown | Total Strategy Return | Annualized Return | Direction Accuracy | Trades |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| linear | n+1 | -0.112 | -0.158 | -99.33% | -87.46% | -71.66% | 51.1% | 415 |
| linear | n+2 | -0.334 | -0.459 | -99.66% | -94.42% | -83.04% | 50.7% | 205 |
| linear | n+3 | -0.175 | -0.232 | -98.22% | -94.56% | -83.21% | 49.0% | 137 |
| linear | n+4 | -1.327 | -1.608 | -99.92% | -99.76% | -97.53% | 48.5% | 103 |
| linear | n+5 | 0.205 | 0.297 | -90.77% | -81.28% | -64.30% | 50.5% | 82 |
| random_forest | n+1 | -1.523 | -1.960 | -99.89% | -99.60% | -96.51% | 48.9% | 415 |
| random_forest | n+2 | -1.310 | -1.667 | -99.96% | -99.54% | -96.34% | 47.8% | 205 |
| random_forest | n+3 | -1.034 | -1.288 | -99.81% | -99.44% | -95.82% | 46.3% | 137 |
| random_forest | n+4 | -0.503 | -0.724 | -99.74% | -96.72% | -87.64% | 47.6% | 103 |
| random_forest | n+5 | -0.100 | -0.146 | -98.88% | -89.59% | -75.11% | 50.0% | 82 |

## Metric definitions

- **Sharpe Ratio:** annualized excess strategy return divided by total return volatility.
- **Sortino Ratio:** annualized excess strategy return divided by downside deviation.
- **Maximum Drawdown:** largest peak-to-trough decline in the compounded strategy equity curve.

## Interpretation and limitations

These are backtested directional-strategy metrics, not guaranteed investment performance. They depend on the prediction timing, target construction, risk-free rate, transaction-cost assumption, and available out-of-sample sample size. Short positions are modeled as the negative of the underlying asset return; borrow fees, funding, market impact, taxes, and execution slippage beyond the configured transaction cost are not modeled.

The every-h-th-row sampling avoids overlapping forecast holding periods within each horizon. The resulting trade sequence is a simplified strategy evaluation and should be interpreted alongside the separate MAE, RMSE, and direction-accuracy forecast comparison report.
