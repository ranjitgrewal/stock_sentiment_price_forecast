from __future__ import annotations

import argparse
import re
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from asset_ingestion.database import connect_database


POSITIVE_WORDS = {
    "beat", "beats", "bull", "bullish", "buy", "growth", "gain", "gains", "good",
    "great", "higher", "improve", "improved", "positive", "profit", "profits", "rally",
    "strong", "surge", "up", "upgrade", "upside", "win", "record", "outperform",
}
NEGATIVE_WORDS = {
    "bear", "bearish", "bad", "below", "crash", "decline", "drop", "downgrade", "fall",
    "fell", "fraud", "lower", "loss", "losses", "negative", "risk", "sell",
    "weak", "weakness", "down", "miss", "misses", "underperform", "lawsuit",
}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "aren't", "don't", "didn't", "won't"}
FEATURE_COLUMNS = (
    "ticker", "market_date", "close", "tweet_count", "sentiment_mean", "sentiment_positive_share",
    "sentiment_negative_share", "sentiment_3d_mean", "sentiment_5d_mean", "return_1d_lag",
    "return_5d_lag", "return_20d_lag", "volatility_5d_lag", "volatility_20d_lag",
    "target_close_n0", "target_close_n1", "target_close_n2", "target_close_n3", "target_close_n4",
    "target_close_n5",
)


def score_text(text: str) -> float:
    tokens = [token.casefold() for token in TOKEN_PATTERN.findall(text or "")]
    if not tokens:
        return 0.0
    polarity = 0
    for index, token in enumerate(tokens):
        direction = 1 if token in POSITIVE_WORDS else -1 if token in NEGATIVE_WORDS else 0
        if direction and any(previous in NEGATORS for previous in tokens[max(0, index - 3):index]):
            direction *= -1
        polarity += direction
    return polarity / len(tokens)


def _daily_post_features(connection: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    cursor = connection.execute(
        "SELECT market_date, text FROM tweets WHERE ticker = ? ORDER BY market_date",
        (ticker.upper(),),
    )
    for row in cursor:
        score = score_text(row["text"])
        aggregate = totals[row["market_date"]]
        aggregate[0] += 1
        aggregate[1] += score
        aggregate[2] += score > 0
        aggregate[3] += score < 0
    return pd.DataFrame(
        [
            {
                "market_date": market_date,
                "tweet_count": int(values[0]),
                "sentiment_mean": values[1] / values[0],
                "sentiment_positive_share": values[2] / values[0],
                "sentiment_negative_share": values[3] / values[0],
            }
            for   market_date, values in totals.items()
        ],
        columns=["market_date", "tweet_count", "sentiment_mean", "sentiment_positive_share", "sentiment_negative_share"],
    )


def build_daily_features(connection: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    prices = pd.read_sql_query(
        "SELECT market_date, close FROM price_bars WHERE ticker = ? ORDER BY market_date",
        connection,
        params=(ticker.upper(),),
    )
    if prices.empty:
        raise ValueError(f"No price bars found for {ticker.upper()}; cannot build daily features")

    tweets = _daily_post_features(connection, ticker)
    daily = prices.merge(tweets, on="market_date", how="left")
    daily["tweet_count"] = daily["tweet_count"].fillna(0).astype("int64")
    for column in ("sentiment_mean", "sentiment_positive_share", "sentiment_negative_share"):
        daily[column] = daily[column].fillna(0.0)

    daily["sentiment_3d_mean"] = daily["sentiment_mean"].rolling(3, min_periods=1).mean()
    daily["sentiment_5d_mean"] = daily["sentiment_mean"].rolling(5, min_periods=1).mean()
    daily["return_1d_lag"] = daily["close"].pct_change(1).shift(1)
    daily["return_5d_lag"] = daily["close"].pct_change(5).shift(1)
    daily["return_20d_lag"] = daily["close"].pct_change(20).shift(1)
    daily["volatility_5d_lag"] = daily["close"].pct_change().rolling(5).std().shift(1)
    daily["volatility_20d_lag"] = daily["close"].pct_change().rolling(20).std().shift(1)
    for horizon in range(6):
        daily[f"target_close_n{horizon}"] = daily["close"].shift(-horizon)

    daily.insert(0, "ticker", ticker.upper())
    return daily.loc[:, FEATURE_COLUMNS]


def persist_daily_features(connection: sqlite3.Connection, features: pd.DataFrame, ticker: str) -> None:
    columns_sql = ", ".join(FEATURE_COLUMNS)
    placeholders = ", ".join("?" for _ in FEATURE_COLUMNS)
    with connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS daily_features (
                ticker TEXT NOT NULL,
                market_date TEXT NOT NULL,
                close REAL,
                tweet_count INTEGER NOT NULL,
                sentiment_mean REAL NOT NULL,
                sentiment_positive_share REAL NOT NULL,
                sentiment_negative_share REAL NOT NULL,
                sentiment_3d_mean REAL NOT NULL,
                sentiment_5d_mean REAL NOT NULL,
                return_1d_lag REAL,
                return_5d_lag REAL,
                return_20d_lag REAL,
                volatility_5d_lag REAL,
                volatility_20d_lag REAL,
                target_close_n0 REAL,
                target_close_n1 REAL,
                target_close_n2 REAL,
                target_close_n3 REAL,
                target_close_n4 REAL,
                target_close_n5 REAL,
                PRIMARY KEY (ticker, market_date)
            )"""
        )
        connection.execute("DELETE FROM daily_features WHERE ticker = ?", (ticker.upper(),))
        records: list[tuple[Any, ...]] = []
        for row in features.itertuples(index=False, name=None):
            records.append(tuple(None if pd.isna(value) else value for value in row))
        connection.executemany(
            f"INSERT INTO daily_features ({columns_sql}) VALUES ({placeholders})",
            records,
        )


def write_sentiment_summary(
    features: pd.DataFrame,
    ticker: str,
    output_path: str | Path,
    data_quality_note: str | None = None,
) -> None:
    recent = features.tail(20)
    post_days = int((recent["tweet_count"] > 0).sum())
    total_posts = int(recent["tweet_count"].sum())
    mean_score = float(recent.loc[recent["tweet_count"] > 0, "sentiment_mean"].mean()) if post_days else 0.0
    if post_days < 5:
        outlook = (
            f"Only {post_days} of the {len(recent)} recent price sessions contain X posts; "
            "coverage is too sparse to assign a reliable current sentiment direction."
        )
        label = "Insufficient X coverage"
    elif mean_score > 0.02:
        outlook = "The lexicon baseline indicates a mildly positive tone in the observed recent X posts. This is a descriptive signal, not a price forecast."
        label = "Mildly positive"
    elif mean_score < -0.02:
        outlook = "The lexicon baseline indicates a mildly negative tone in the observed recent X posts. This is a descriptive signal, not a price forecast."
        label = "Mildly negative"
    else:
        outlook = "The lexicon baseline is approximately neutral across the observed recent X posts. This is a descriptive signal, not a price forecast."
        label = "Approximately neutral"

    latest = features.iloc[-1]
    start_close = float(recent.iloc[0]["close"])
    end_close = float(recent.iloc[-1]["close"])
    price_change = (end_close / start_close - 1.0) if start_close else 0.0
    source_start = str(recent.iloc[0]["market_date"])
    source_end = str(recent.iloc[-1]["market_date"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"# {ticker.upper()} X Sentiment Outlook",
                "",
                f"**As of:** {source_end}  ",
                f"**Baseline outlook:** {label}",
                "",
                "## Recent evidence",
                f"- Window: {source_start} to {source_end} ({len(recent)} available price sessions).",
                f"- X posts: {total_posts} across {post_days} sessions with posts.",
                f"- Mean lexicon score on post days: {mean_score:+.4f} (positive words count +1, negative words -1, divided by token count).",
                f"- Price change across the available session window: {price_change:+.2%}.",
                f"- Latest close: {float(latest['close']):.2f}.",
                "",
                "## Interpretation and limits",
                outlook,
                "",
                *(
                    ["## Pipeline data coverage and impact", data_quality_note, ""]
                    if data_quality_note
                    else []
                ),
                "## How to read this report",
                "This report is a descriptive snapshot of recent X sentiment around the ticker, not a buy, sell, or price forecast. The score is calculated with a simple lexicon: positive words increase the score, negative words decrease it, and the total is normalized by the number of tokens in the observed posts.",
                "",
                "The price move in the recent window is shown separately so you can compare the social tone against actual market behavior, but the model does not claim that social chatter alone explains price action. When there are only a few post-bearing sessions, the correct conclusion is uncertainty, not a strong bullish or bearish call.",
                "",
                "The sentiment score is a transparent word-list baseline, not a finance-trained language model. It does not account for sarcasm, context, ticker ambiguity, bots, or post reach. Days without posts are assigned a neutral score for the feature table but are not treated as observed neutral sentiment in this summary. News is stored separately and is not used in sentiment features or forecasts.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily X sentiment/price features and a one-page outlook.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--database", default="data/market_data.sqlite3")
    parser.add_argument("--summary", help="Summary Markdown output path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ticker = args.ticker.upper()
    summary_path = args.summary or f"reports/{ticker.lower()}_sentiment_summary.md"
    connection = connect_database(args.database)
    try:
        features = build_daily_features(connection, ticker)
        persist_daily_features(connection, features, ticker)
    finally:
        connection.close()
    write_sentiment_summary(features, ticker, summary_path)
    print(f"Built {len(features)} daily feature rows for {ticker}; summary: {summary_path}")


if __name__ == "__main__":
    main()