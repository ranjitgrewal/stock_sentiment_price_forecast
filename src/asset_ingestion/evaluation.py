from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from asset_ingestion.model import (
    MODEL_FEATURE_COLUMNS,
    fit_linear_model,
    fit_random_forest_model,
    prepare_model_frame,
)


def _out_of_sample_predictions(
    features: pd.DataFrame,
    horizon: int,
    model_type: str,
    train_window: int,
    validation_window: int = 5,
    min_training_rows: int = 10,
) -> pd.DataFrame:
    """Recreate the model module's rolling-origin validation and retain predictions."""
    if model_type not in {"linear", "random_forest"}:
        raise ValueError("model_type must be 'linear' or 'random_forest'")

    model_df = prepare_model_frame(features, horizon)
    effective_min = min(min_training_rows, max(1, train_window))

    if len(model_df) < effective_min + validation_window:
        raise ValueError(
            f"Not enough rows to evaluate {model_type}, horizon n+{horizon}: "
            f"{len(model_df)} rows"
        )

    effective_window = max(1, min(train_window, len(model_df) - validation_window))
    prediction_batches: list[pd.DataFrame] = []

    for start in range(0, len(model_df) - effective_window, validation_window):
        train_df = model_df.iloc[start : start + effective_window].copy()
        valid_start = start + effective_window
        valid_end = valid_start + validation_window
        if valid_end > len(model_df):
            break

        valid_df = model_df.iloc[valid_start:valid_end].copy()
        if len(train_df) < effective_min or valid_df.empty:
            continue

        train_X = train_df[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)
        train_y = train_df[f"target_close_n{horizon}"].to_numpy(dtype=float)
        valid_X = valid_df[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype=float)

        if model_type == "linear":
            coefficients = fit_linear_model(train_X, train_y)
            predictions = (
                np.column_stack([np.ones(valid_X.shape[0]), valid_X])
                @ coefficients
            )
        else:
            fitted_model = fit_random_forest_model(train_X, train_y)
            predictions = fitted_model.predict(valid_X)

        batch = valid_df[["market_date", "close", f"target_close_n{horizon}"]].copy()
        batch["predicted_close"] = predictions
        batch["actual_close"] = batch[f"target_close_n{horizon}"]
        batch["row_position"] = np.arange(valid_start, valid_end)
        prediction_batches.append(batch)

    if not prediction_batches:
        raise ValueError(
            f"No out-of-sample predictions for {model_type}, horizon n+{horizon}"
        )

    return (
        pd.concat(prediction_batches, ignore_index=True)
        .sort_values("row_position")
        .drop_duplicates(subset=["row_position"])
        .reset_index(drop=True)
    )


def _calculate_risk_metrics(
    trade_returns: pd.Series,
    risk_free_rate: float,
    periods_per_year: float,
) -> dict[str, float | int]:
    returns = pd.to_numeric(trade_returns, errors="coerce").dropna()
    if returns.empty:
        raise ValueError("No valid strategy returns available for risk metrics")

    periodic_rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - periodic_rf

    std = excess.std(ddof=1)
    sharpe = (
        float(np.sqrt(periods_per_year) * excess.mean() / std)
        if pd.notna(std) and std > 0
        else float("nan")
    )

    downside = np.minimum(excess.to_numpy(dtype=float), 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))))
    sortino = (
        float(np.sqrt(periods_per_year) * excess.mean() / downside_deviation)
        if downside_deviation > 0
        else float("nan")
    )

    equity = (1.0 + returns).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0

    total_return = float(equity.iloc[-1] - 1.0)
    annualized_return = (
        float(equity.iloc[-1] ** (periods_per_year / len(returns)) - 1.0)
        if equity.iloc[-1] > 0
        else float("nan")
    )

    return {
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": float(drawdown.min()),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "trade_observations": int(len(returns)),
    }


def evaluate_model_risk(
    features: pd.DataFrame,
    horizon: int,
    model_type: str,
    train_window: int,
    risk_free_rate: float = 0.02,
    one_way_transaction_cost: float = 0.001,
    validation_window: int = 5,
) -> dict[str, Any]:
    """
    Evaluate out-of-sample directional trades for one model/horizon.

    For horizon n+h, trades are sampled every h observations so holding
    periods do not overlap. Costs are charged on entry and exit. The
    resulting risk metrics describe this simplified long/short strategy,
    not the model's raw price-prediction error.
    """
    if not 1 <= horizon <= 5:
        raise ValueError("Risk evaluation supports horizons n+1 through n+5")
    if not 0 <= risk_free_rate < 1:
        raise ValueError("risk_free_rate must be between 0 and 1")
    if not 0 <= one_way_transaction_cost < 1:
        raise ValueError("one_way_transaction_cost must be between 0 and 1")

    predictions = _out_of_sample_predictions(
        features=features,
        horizon=horizon,
        model_type=model_type,
        train_window=train_window,
        validation_window=validation_window,
        min_training_rows=max(5, min(20, train_window // 2)),
    )

    predictions["predicted_direction"] = np.sign(
        predictions["predicted_close"] - predictions["close"]
    )
    predictions["actual_direction"] = np.sign(
        predictions["actual_close"] - predictions["close"]
    )
    predictions["direction_correct"] = (
        predictions["predicted_direction"]
        == predictions["actual_direction"]
    )

    # A trade opened at the forecast origin is held for h trading sessions.
    # Select every h-th forecast origin to avoid overlapping holding periods.
    trades = predictions.iloc[::horizon].copy()
    trades["asset_return"] = (
        trades["actual_close"] / trades["close"] - 1.0
    )
    trades["strategy_return"] = (
        trades["predicted_direction"] * trades["asset_return"]
        - 2.0 * one_way_transaction_cost
    )

    metrics = _calculate_risk_metrics(
        trades["strategy_return"],
        risk_free_rate=risk_free_rate,
        periods_per_year=252.0 / horizon,
    )
    metrics.update(
        {
            "model": model_type,
            "horizon": horizon,
            "forecast_observations": int(len(predictions)),
            "direction_accuracy": float(predictions["direction_correct"].mean()),
            "mean_absolute_error": float(
                np.mean(
                    np.abs(
                        predictions["predicted_close"]
                        - predictions["actual_close"]
                    )
                )
            ),
            "one_way_transaction_cost": one_way_transaction_cost,
            "risk_free_rate": risk_free_rate,
        }
    )
    return metrics


def write_risk_evaluation_report(
    ticker: str,
    features: pd.DataFrame,
    output_path: str | Path,
    train_window: int | None = None,
    risk_free_rate: float = 0.02,
    one_way_transaction_cost: float = 0.001,
) -> list[dict[str, Any]]:
    """Write Sharpe, Sortino and drawdown metrics to a Markdown report."""
    if train_window is None:
        train_window = max(20, min(60, len(features) // 2))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for model_type in ("linear", "random_forest"):
        for horizon in range(1, 6):
            try:
                results.append(
                    evaluate_model_risk(
                        features=features,
                        horizon=horizon,
                        model_type=model_type,
                        train_window=train_window,
                        risk_free_rate=risk_free_rate,
                        one_way_transaction_cost=one_way_transaction_cost,
                    )
                )
            except ValueError as exc:
                errors.append(f"{model_type} n+{horizon}: {exc}")

    lines = [
        f"# {ticker.upper()} Trading Risk Evaluation",
        "",
        "## Methodology",
        "",
        "Risk metrics are calculated from chronological out-of-sample predictions "
        "generated with the same feature columns, model fitting functions, rolling "
        "training window, and five-row validation batches used by the forecast model.",
        "",
        "For each forecast horizon n+h (h=1..5), the strategy takes a long position "
        "when predicted future close is above the current close, a short position "
        "when it is below, and no position when equal. Realized trade return is the "
        "signed return from current close to the actual n+h close. To avoid overlapping "
        "holding periods, the report uses every h-th out-of-sample forecast. Sharpe "
        "and Sortino are annualized using 252/h periods per year. Entry and exit costs "
        "are deducted using the configured one-way transaction cost.",
        "",
        f"- Annual risk-free rate: {risk_free_rate:.2%}",
        f"- One-way transaction cost: {one_way_transaction_cost:.3%}",
        f"- Training window: {train_window} rows",
        "- Validation window: 5 rows",
        "- Horizons included: n+1 through n+5 (n+0 excluded)",
        "",
        "## Risk and strategy performance",
        "",
        "| Model | Horizon | Sharpe Ratio | Sortino Ratio | Max Drawdown | Total Strategy Return | Annualized Return | Direction Accuracy | Trades |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    for item in results:
        lines.append(
            f"| {item['model']} | n+{item['horizon']} "
            f"| {item['sharpe_ratio']:.3f} "
            f"| {item['sortino_ratio']:.3f} "
            f"| {item['max_drawdown']:.2%} "
            f"| {item['total_return']:.2%} "
            f"| {item['annualized_return']:.2%} "
            f"| {item['direction_accuracy']:.1%} "
            f"| {item['trade_observations']} |"
        )

    if errors:
        lines.extend(["", "## Horizons not evaluated", ""])
        lines.extend(f"- {message}" for message in errors)

    lines.extend(
        [
            "",
            "## Metric definitions",
            "",
            "- **Sharpe Ratio:** annualized excess strategy return divided by total return volatility.",
            "- **Sortino Ratio:** annualized excess strategy return divided by downside deviation.",
            "- **Maximum Drawdown:** largest peak-to-trough decline in the compounded strategy equity curve.",
            "",
            "## Interpretation and limitations",
            "",
            "These are backtested directional-strategy metrics, not guaranteed investment "
            "performance. They depend on the prediction timing, target construction, "
            "risk-free rate, transaction-cost assumption, and available out-of-sample "
            "sample size. Short positions are modeled as the negative of the underlying "
            "asset return; borrow fees, funding, market impact, taxes, and execution "
            "slippage beyond the configured transaction cost are not modeled.",
            "",
            "The every-h-th-row sampling avoids overlapping forecast holding periods "
            "within each horizon. The resulting trade sequence is a simplified strategy "
            "evaluation and should be interpreted alongside the separate MAE, RMSE, and "
            "direction-accuracy forecast comparison report.",
            "",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")
    return results
