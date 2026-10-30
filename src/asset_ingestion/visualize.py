from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from asset_ingestion.database import connect_database


def generate_sentiment_price_chart(features: pd.DataFrame, output_path: str | Path) -> Path:
    """Create a two-panel chart showing sentiment and close price over time."""
    if features.empty:
        raise ValueError("No feature rows available to visualize")

    chart_df = features.copy().sort_values("market_date").reset_index(drop=True)
    required = {"market_date", "close"}
    missing = sorted(required - set(chart_df.columns))
    if missing:
        raise ValueError(f"Missing required chart columns: {missing}")

    sentiment_col = "sentiment_mean" if "sentiment_mean" in chart_df.columns else "sentiment_3d_mean"
    if sentiment_col not in chart_df.columns:
        raise ValueError("Missing sentiment column for charting")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    ticker = str(chart_df.iloc[0].get("ticker", "ASSET")).upper()
    dates = pd.to_datetime(chart_df["market_date"])
    close_values = pd.to_numeric(chart_df["close"], errors="coerce")
    sentiment_values = pd.to_numeric(chart_df[sentiment_col], errors="coerce").fillna(0.0)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, height_ratios=[2, 1])

    axes[0].plot(dates, close_values, color="tab:blue", linewidth=2.5, label="Close")
    axes[0].set_ylabel("Price")
    axes[0].set_title(f"{ticker} sentiment vs. price over time")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(dates, sentiment_values, color="tab:orange", linewidth=2.5, label="Sentiment mean")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set_ylabel("Lexicon sentiment")
    axes[1].set_xlabel("Market date")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")

    for axis in axes:
        axis.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output


def load_feature_table(connection: Any, ticker: str) -> pd.DataFrame:
    features = pd.read_sql_query(
        "SELECT * FROM daily_features WHERE ticker = ? ORDER BY market_date",
        connection,
        params=(ticker.upper(),),
    )
    if features.empty:
        raise ValueError(f"No daily features found for {ticker.upper()}")
    return features


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a sentiment-versus-price chart from the daily feature table.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol to chart")
    parser.add_argument("--database", default="data/market_data.sqlite3", help="SQLite database path")
    parser.add_argument("--output", default=None, help="PNG output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = args.output or Path("reports") / f"{args.ticker.lower()}_sentiment_vs_price.png"
    connection = connect_database(args.database)
    try:
        features = load_feature_table(connection, args.ticker)
    finally:
        connection.close()

    chart_path = generate_sentiment_price_chart(features, output_path)
    print(f"Sentiment chart saved to {chart_path}")


if __name__ == "__main__":
    main()
