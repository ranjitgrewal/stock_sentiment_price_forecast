# X Sentiment Price Pipeline

Python pipeline for collecting recent X posts, market prices, and news; engineering sentiment and price features; evaluating forecasts; and generating reports and charts.

The pipeline compares Linear Regression and Random Forest models using chronological walk-forward validation. It also evaluates a simplified directional trading strategy using risk-adjusted performance metrics, including Sharpe Ratio, Sortino Ratio, and Maximum Drawdown.

News is stored separately and is not used as a model input.

## Setup

Use Python 3.10 or newer:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
cp .env.example .env
```

Set valid `APIFY_API_TOKEN` and `TAVILY_API_KEY` values in `.env`.

The default X Actor is `apidojo/tweet-scraper`; the price Actor is `pink_comic/yahoo-finance-stock-data`. Actor IDs and input JSON can be changed in `.env`.

Live requests can incur charges. Review current Actor pricing and account limits first.

## Ingest recent data

The ingestion command fetches X posts, daily prices, and recent news for the same lookback window. The assignment-compliant default is 180 days; supported windows are 30, 60, 90, or 180 days.

```sh
.venv/bin/python -m asset_ingestion --ticker AMZN --keyword Amazon --days 60 --database src/data/market_data.sqlite3
```

Set `--max-items` to override the X Actor result cap for one run.

The default market timezone is `America/New_York`; use `--market-timezone UTC` for UTC-based assets such as Bitcoin.

## Build features and sentiment summary

Features are built from stored tweets and price bars. They include:

* Post volume
* Lexicon-based sentiment scores
* Positive and negative sentiment shares
* Rolling sentiment averages
* Lagged price returns
* Rolling price volatility
* Forward price targets from `n+0` through `n+5`

```sh
.venv/bin/python -m asset_ingestion.features --ticker AMZN --database src/data/market_data.sqlite3 --summary src/reports/amzn_sentiment_summary.md
```

Sentiment uses a small transparent English word list, not a finance-trained model. Days without posts are represented as neutral in the feature table, but are not treated as observed neutral sentiment in the summary.

## Forecast report and chart

One model command creates the side-by-side forecast comparison report and the sentiment-versus-price PNG:

```sh
.venv/bin/python -m asset_ingestion.model --ticker AMZN --database src/data/market_data.sqlite3 --output src/reports/amzn_forecast_comparison.md
```

The report compares Linear Regression and Random Forest for horizons `n+0` to `n+5`, using identical chronological walk-forward validation blocks.

It includes:

* Model comparison table
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Directional accuracy
* Short results summary

The chart is written beside the report by default; override it with `--chart-output`.

Lower MAE and RMSE indicate smaller prediction errors. Direction accuracy near 50% is approximately chance-level. `n+0` is the current-period close, not a future direction forecast.

## Risk evaluation

The evaluation module calculates risk and return metrics using out-of-sample predictions generated through chronological walk-forward validation.

It evaluates both Linear Regression and Random Forest across future forecast horizons `n+1` through `n+5`.

```python
from asset_ingestion.evaluation import write_risk_evaluation_report

write_risk_evaluation_report(
    ticker="AMZN",
    features=features,
    output_path="src/reports/amzn_risk_evaluation.md",
)
```

The evaluation report includes:

| Metric            | Description                                                                             |
| ----------------- | --------------------------------------------------------------------------------------- |
| Sharpe Ratio      | Measures return relative to overall return volatility, adjusted for the risk-free rate. |
| Sortino Ratio     | Measures return relative to downside deviation.                                         |
| Maximum Drawdown  | Measures the largest peak-to-trough decline in strategy equity.                         |
| Total Return      | Cumulative return over the evaluated out-of-sample period.                              |
| Annualized Return | Return expressed on an annualized basis using the configured trading-period assumption. |
| Number of Trades  | Number of evaluated strategy positions.                                                 |

### Strategy assumptions

The evaluation uses a simplified directional strategy:

* Long position when the predicted future close is above the current close.
* Short position when the predicted future close is below the current close.
* Flat position when the prediction does not indicate a directional move.
* Uses actual future price movement to calculate the return of each predicted position.
* Samples forecasts at the selected horizon interval to reduce overlapping holding periods.
* Applies estimated transaction costs when entering and exiting positions.

Default evaluation assumptions:

| Parameter                |                                                     Default |
| ------------------------ | ----------------------------------------------------------: |
| Risk-free rate           |                                                          2% |
| One-way transaction cost |                                                        0.1% |
| Validation window        |                                              5 observations |
| Annualization basis      | 252 trading periods per year, adjusted for forecast horizon |

These are configurable evaluation assumptions, not observed market costs or guaranteed trading conditions.

**Important:** These risk metrics describe a simplified simulated directional strategy, not actual portfolio performance. The current evaluation does not fully model borrow fees, funding costs, market impact, taxes, or execution slippage beyond the configured transaction cost. Results should be interpreted as historical out-of-sample research metrics rather than a guarantee of future returns.

The `n+0` horizon is excluded from risk evaluation because it represents the current-period close rather than a future trading forecast.

## Run the full pipeline

The full pipeline performs ingestion, feature generation, forecasting, risk evaluation, and report/chart generation.

```sh
.venv/bin/python -m asset_ingestion.pipeline --ticker AAPL --keyword Amazon --days 60 --database src/data/market_data.sqlite3 --output-dir src/reports
```

The historical tweet CSV is optional. To supplement the scrape with a local historical tweet file, add:

```sh
--data-file src/data/historical_tweets.csv
```

If tweet dates do not overlap price dates, the pipeline displays the date ranges and explains the impact, then asks for another CSV, `P` to proceed, or `Q` to quit.

An alternate CSV is imported before coverage is checked again and downstream features and reports are regenerated.

`--proceed-without-overlap` skips the interactive prompt. Reports disclose when sentiment did not contribute and model inputs were price-derived only.

`--skip-live-ingestion` avoids API requests and uses data already in the database.

### Pipeline outputs

The full pipeline generates the following files in the selected output directory:

| Output                            | Description                                             |
| --------------------------------- | ------------------------------------------------------- |
| `<ticker>_forecast_comparison.md` | Linear Regression and Random Forest forecast comparison |
| `<ticker>_risk_evaluation.md`     | Sharpe, Sortino, drawdown, and return metrics           |
| `<ticker>_sentiment_summary.md`   | Sentiment and post-volume summary                       |
| `<ticker>_sentiment_vs_price.png` | Sentiment-versus-price chart                            |

Example output directory:

```text
src/reports/
├── aapl_forecast_comparison.md
├── aapl_risk_evaluation.md
├── aapl_sentiment_summary.md
└── aapl_sentiment_vs_price.png
```

## Import historical posts

The local `src/data/Tweet.csv` has the following fields:

* `tweet_id`
* `writer`
* `post_date` (Unix seconds)
* `body`
* `comment_num`
* `retweet_num`
* `like_num`

Import the mapped AMZN posts without API requests:

```sh
.venv/bin/python -m asset_ingestion.import_history --ticker AMZN --file src/data/Tweet.csv --company-tweet-file src/data/Company_Tweet.csv --database src/data/market_data.sqlite3
```

The importer streams CSV and JSONL files, filters by terms or exact company mapping, and generates a stable ID for historical rows that lack one.

Reimports update matching records. Posts contribute to daily features only on dates with matching price bars.

Bitcoin archives in `src/data/Bitcoin_tweets.csv` and `src/data/Bitcoin_tweets_dataset_2.csv` cover 2021–2023. They can be imported under `BTC-USD`, but matching BTC price bars are required for those dates before the posts can contribute to features or forecasts.

The multi-million-row CSVs take time and disk space to import. Respect each dataset's license and attribution requirements.

## Database tables

* `tweets`: normalized post text, UTC timestamp, engagement, follower count, and source JSON.
* `price_bars`: daily OHLCV price bars.
* `news`: Tavily title, URL, content, publication date, and relevance.
* `daily_alignment`: price dates with joined tweet-engagement and news aggregates.
* `daily_features`: sentiment/price model inputs and forward price targets.

The active database used in the project examples is `src/data/market_data.sqlite3`.

## Project structure

The relevant package modules include:

```text
src/
└── asset_ingestion/
    ├── model.py
    ├── evaluation.py
    ├── pipeline.py
    ├── features.py
    └── import_history.py
```

`evaluation.py` contains the out-of-sample risk evaluation logic and Markdown report generation. `pipeline.py` invokes the evaluation module after generating the forecast comparison.

## Tests

Run the ingestion tests:

```sh
.venv/bin/python -m pytest tests/test_ingestion.py -q
```

Run the complete test suite:

```sh
.venv/bin/python -m pytest -q
```
