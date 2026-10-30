from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from asset_ingestion.database import connect_database
from asset_ingestion.visualize import generate_sentiment_price_chart

MODEL_FEATURE_COLUMNS = (
    "tweet_count",
    "sentiment_mean",
    "sentiment_positive_share",
    "sentiment_negative_share",
    "sentiment_3d_mean",
    "sentiment_5d_mean",
    "return_1d_lag",
    "return_5d_lag",
    "return_20d_lag",
    "volatility_5d_lag",
    "volatility_20d_lag",
)


def prepare_model_frame(features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if not 0 <= horizon <= 5:
        raise ValueError("horizon must be between 0 and 5")

    model_df = features.copy().sort_values("market_date").reset_index(drop=True)
    target_col = f"target_close_n{horizon}"
    required = list(MODEL_FEATURE_COLUMNS) + [target_col, "close"]
    missing = [column for column in required if column not in model_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for horizon {horizon}: {missing}")

    model_df = model_df.copy()
    for column in list(MODEL_FEATURE_COLUMNS):
        model_df[column] = pd.to_numeric(model_df[column], errors="coerce").fillna(0.0)
    model_df = model_df.dropna(subset=[target_col, "close"]).copy()
    if model_df.empty:
        raise ValueError(f"No valid rows available for horizon {horizon}")
    return model_df


def prepare_model_data(features: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return model inputs and target values for one forecasting horizon."""
    model_df = prepare_model_frame(features, horizon)
    X = model_df[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
    y = model_df[f"target_close_n{horizon}"].to_numpy(dtype=float)
    return X, y


def fit_linear_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(X.shape[0]), X])
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coeffs


def fit_random_forest_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=100,
        max_features=0.8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    return model.fit(X, y)


def walk_forward_validation(
    features: pd.DataFrame,
    horizon: int,
    train_window: int = 60,
    validation_window: int = 1,
    min_training_rows: int = 10,
    model_type: str = "linear",
) -> dict[str, Any]:
    """Evaluate one model with rolling-origin validation for a forecasting target."""
    if model_type not in {"linear", "random_forest"}:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    model_df = prepare_model_frame(features, horizon)
    effective_min_training_rows = min(min_training_rows, max(1, train_window))
    if len(model_df) < effective_min_training_rows + validation_window:
        raise ValueError(f"Not enough rows to validate horizon {horizon}: {len(model_df)}")

    effective_window = max(1, min(train_window, len(model_df) - validation_window))
    fold_metrics: list[dict[str, float]] = []
    for start in range(0, len(model_df) - effective_window, validation_window):
        train_df = model_df.iloc[start : start + effective_window].copy()
        valid_start = start + effective_window
        valid_end = valid_start + validation_window
        if valid_end > len(model_df):
            break
        valid_df = model_df.iloc[valid_start:valid_end].copy()
        if len(train_df) < effective_min_training_rows or valid_df.empty:
            continue

        train_X = train_df[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
        train_y = train_df[f"target_close_n{horizon}"].to_numpy(dtype=float)
        valid_X = valid_df[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
        valid_y = valid_df[f"target_close_n{horizon}"].to_numpy(dtype=float)
        valid_close = valid_df["close"].to_numpy(dtype=float)

        if model_type == "linear":
            coeffs = fit_linear_model(train_X, train_y)
            preds = np.column_stack([np.ones(valid_X.shape[0]), valid_X]) @ coeffs
        else:
            model = fit_random_forest_model(train_X, train_y)
            preds = model.predict(valid_X)
        errors = preds - valid_y
        abs_errors = np.abs(errors)
        direction_signal = np.sign(preds - valid_close)
        actual_signal = np.sign(valid_y - valid_close)
        direction_matches = (direction_signal == actual_signal).astype(float)

        fold_metrics.append(
            {
                "fold": len(fold_metrics),
                "train_rows": int(len(train_df)),
                "valid_rows": int(len(valid_df)),
                "mae": float(np.mean(abs_errors)),
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "direction_accuracy": float(np.mean(direction_matches)),
            }
        )

    if not fold_metrics:
        raise ValueError(f"No walk-forward folds were created for horizon {horizon}")

    aggregate = {
        "model": model_type,
        "horizon": int(horizon),
        "target_column": f"target_close_n{horizon}",
        "n_folds": len(fold_metrics),
        "n_observations": int(sum(metric["valid_rows"] for metric in fold_metrics)),
        "mae": float(np.mean([metric["mae"] for metric in fold_metrics])),
        "rmse": float(np.mean([metric["rmse"] for metric in fold_metrics])),
        "direction_accuracy": float(np.mean([metric["direction_accuracy"] for metric in fold_metrics])),
        "fold_metrics": fold_metrics,
    }
    return aggregate


def load_feature_table(connection: Any, ticker: str) -> pd.DataFrame:
    ticker_upper = ticker.upper()
    features = pd.read_sql_query(
        "SELECT * FROM daily_features WHERE ticker = ? ORDER BY market_date",
        connection,
        params=(ticker_upper,),
    )
    if features.empty:
        raise ValueError(f"No daily features found for {ticker_upper}")
    return features


def build_horizon_summary(features: pd.DataFrame, train_window: int | None = None) -> pd.DataFrame:
    if train_window is None:
        train_window = max(20, min(60, len(features) // 2))

    rows = []
    for model_type in ("linear", "random_forest"):
        for horizon in range(6):
            min_training_rows = max(5, min(20, train_window // 2))
            metrics = walk_forward_validation(
                features,
                horizon=horizon,
                train_window=train_window,
                validation_window=5,
                min_training_rows=min_training_rows,
                model_type=model_type,
            )
            rows.append(
                {
                    "model": model_type,
                    "horizon": horizon,
                    "target_column": metrics["target_column"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "direction_accuracy": metrics["direction_accuracy"],
                    "n_observations": metrics["n_observations"],
                    "n_folds": metrics["n_folds"],
                }
            )
    return pd.DataFrame(rows)


def write_model_summary(
    ticker: str,
    summary: pd.DataFrame,
    output_path: str | Path,
    data_quality_note: str | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metrics = summary.pivot(index="horizon", columns="model")
    horizons = sorted(int(horizon) for horizon in summary["horizon"].unique())
    table_lines = [
        "| Horizon | Linear MAE | Random Forest MAE | Linear RMSE | Random Forest RMSE | Linear Direction | Random Forest Direction | Folds |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for horizon in horizons:
        values: list[str] = []
        for metric, is_percentage in (
            ("mae", False),
            ("rmse", False),
            ("direction_accuracy", True),
        ):
            for model_name in ("linear", "random_forest"):
                value = metrics[metric].get(model_name, pd.Series(dtype=float)).get(horizon, float("nan"))
                if pd.isna(value):
                    values.append("N/A")
                elif is_percentage:
                    values.append(f"{float(value):.1%}")
                else:
                    values.append(f"{float(value):.2f}")

        fold_counts = []
        for model_name in ("linear", "random_forest"):
            model_rows = summary.loc[
                (summary["horizon"] == horizon) & (summary["model"] == model_name), "n_folds"
            ]
            fold_counts.append(str(int(model_rows.iloc[0])) if not model_rows.empty else "N/A")

        table_lines.append(
            f"| n+{horizon} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | "
            f"{values[4]} | {values[5]} | {fold_counts[0]} / {fold_counts[1]} |"
        )

    future = summary.loc[summary["horizon"].between(1, 5)]
    future_mae = future.pivot(index="horizon", columns="model", values="mae")
    future_rmse = future.pivot(index="horizon", columns="model", values="rmse")
    future_direction = future.pivot(index="horizon", columns="model", values="direction_accuracy")
    paired_horizons = future_mae.dropna(subset=["linear", "random_forest"]).index
    if len(paired_horizons):
        linear_mae = float(future_mae.loc[paired_horizons, "linear"].mean())
        forest_mae = float(future_mae.loc[paired_horizons, "random_forest"].mean())
        linear_rmse = float(future_rmse.loc[paired_horizons, "linear"].mean())
        forest_rmse = float(future_rmse.loc[paired_horizons, "random_forest"].mean())
        linear_direction = float(future_direction.loc[paired_horizons, "linear"].mean())
        forest_direction = float(future_direction.loc[paired_horizons, "random_forest"].mean())
        mae_wins = int(
            (future_mae.loc[paired_horizons, "random_forest"] < future_mae.loc[paired_horizons, "linear"]).sum()
        )
        if forest_mae < linear_mae:
            error_result = "Random Forest had the lower mean MAE"
        elif linear_mae < forest_mae:
            error_result = "Linear regression had the lower mean MAE"
        else:
            error_result = "Both models had the same mean MAE"
        results_summary = (
            f"Across n+1 to n+5, {error_result} over {len(paired_horizons)} paired horizons "
            f"({forest_mae:.2f} vs. {linear_mae:.2f}); it had lower MAE in {mae_wins} of those horizons. "
            f"Mean RMSE was {forest_rmse:.2f} for Random Forest and {linear_rmse:.2f} for linear regression. "
            f"Mean direction accuracy was {forest_direction:.1%} and {linear_direction:.1%}, respectively."
        )
    else:
        results_summary = "There are not enough paired n+1 to n+5 results to summarize the models."

    markdown = [
        f"# {ticker.upper()} Forecast Model Comparison",
        "",
        "## Results by horizon",
        "",
        *table_lines,
        "",
        "Folds are shown as linear / Random Forest. MAE and RMSE are in the ticker's price units; lower is better. Direction accuracy measures whether the predicted change has the same sign as the actual change.",
        "",
        "## Summary",
        "",
        results_summary,
        "",
        *(
            ["## Pipeline data coverage and impact", data_quality_note, ""]
            if data_quality_note
            else []
        ),
        "n+0 is the current-period close, not a future horizon, so its direction score is not evidence of future directional skill. Judge future performance using n+1 through n+5 and chronological out-of-sample results.",
        "",
        "### Interpretation",
        "",
        "The report compares linear regression with a Random Forest using only engineered sentiment and price features. Each model and horizon uses the same rolling-origin folds, keeping validation chronological and preventing future information from entering training data.",
        "",
        "The Random Forest uses 100 trees, a minimum leaf size of four, and a fixed seed to limit variance and make results reproducible. It can learn nonlinear relationships, but added complexity does not guarantee better forecasts. Direction accuracy near 50% is approximately chance-level and should not be treated as trading evidence without substantially more out-of-sample data.",
        "",
    ]
    path.write_text("\n".join(markdown), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward validation for n+0 through n+5 forecasting on daily features.")
    parser.add_argument("--ticker", required=True, help="Ticker to evaluate")
    parser.add_argument("--database", default="data/market_data.sqlite3", help="SQLite database path")
    parser.add_argument("--output", default=None, help="Markdown report path")
    parser.add_argument("--chart-output", default=None, help="PNG chart path; defaults beside the Markdown report")
    parser.add_argument("--train-window", type=int, default=None, help="Rolling training window length to use")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    connection = connect_database(args.database)
    try:
        features = load_feature_table(connection, args.ticker)
    finally:
        connection.close()

    summary = build_horizon_summary(features, train_window=args.train_window)
    report_path = args.output or Path("reports") / f"{args.ticker.lower()}_forecast_summary.md"
    chart_path = args.chart_output or Path(report_path).parent / f"{args.ticker.lower()}_sentiment_vs_price.png"
    write_model_summary(args.ticker, summary, report_path)
    generate_sentiment_price_chart(features, chart_path)
    print(summary.to_string(index=False))
    print(f"Forecast summary saved to {report_path}")
    print(f"Sentiment/price chart saved to {chart_path}")


if __name__ == "__main__":
    main()
