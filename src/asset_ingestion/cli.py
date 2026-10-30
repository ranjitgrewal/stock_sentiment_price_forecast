from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone

from asset_ingestion.database import connect_database, persist_ingestion
from asset_ingestion.normalize import normalize_news, normalize_price_bars, normalize_tweet
from asset_ingestion.providers import (
    fetch_news,
    fetch_price_history,
    fetch_x_posts,
    load_environment,
    required_environment,
)


def ingest(
    ticker: str,
    keywords: list[str],
    database_path: str,
    market_timezone: str,
    apify_actor_id: str,
    apify_price_actor_id: str,
    apify_api_token: str,
    tavily_api_key: str,
    actor_input_json: str | None = None,
    price_actor_input_json: str | None = None,
    end_date: date | None = None,
    lookback_days: int = 180,
) -> tuple[int, int, int]:
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")
    end_date = end_date or (datetime.now(timezone.utc).date() + timedelta(days=1))
    start_date = end_date - timedelta(days=lookback_days)

    post_records = fetch_x_posts(
        ticker, keywords, start_date, end_date, apify_actor_id, apify_api_token, actor_input_json
    )
    price_records = fetch_price_history(
        ticker,
        start_date,
        end_date,
        apify_price_actor_id,
        apify_api_token,
        price_actor_input_json or "",
    )
    news_records = fetch_news(ticker, tavily_api_key, start_date)

    tweets = [
        item
        for record in post_records
        if (item := normalize_tweet(record, ticker, market_timezone)) is not None
        and start_date.isoformat() <= item["market_date"] < end_date.isoformat()
    ]
    bars = [
        bar
        for bar in normalize_price_bars(price_records, ticker, market_timezone)
        if start_date.isoformat() <= bar["market_date"] < end_date.isoformat()
    ]
    retrieved_at = datetime.now(timezone.utc)
    news_items = [
        normalize_news(record, ticker, market_timezone, retrieved_at)
        for record in news_records
        if record.get("url")
    ]

    connection = connect_database(database_path)
    try:
        persist_ingestion(connection, ticker, tweets, bars, news_items)
    finally:
        connection.close()
    return len(tweets), len(bars), len(news_items)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest X posts, historical prices, and news for one asset.")
    parser.add_argument("--ticker", help="Asset ticker or symbol; prompted when omitted")
    parser.add_argument("--keyword", action="append", default=[], help="Additional X search term; repeat as needed")
    parser.add_argument(
        "--max-items",
        type=_positive_int,
        help="Override the X Actor's maxItems for this run only; may increase Apify charges",
    )
    parser.add_argument(
        "--days",
        type=int,
        choices=(30, 60, 90, 180),
        default=180,
        help="Lookback for X posts, prices, and news; defaults to assignment-required 180 days",
    )
    parser.add_argument("--database", default=os.getenv("DATABASE_PATH", "data/market_data.sqlite3"))
    parser.add_argument("--market-timezone", default=os.getenv("MARKET_TIMEZONE", "America/New_York"))
    return parser.parse_args()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _override_max_items(actor_input_json: str | None, max_items: int | None) -> str | None:
    if max_items is None:
        return actor_input_json
    actor_input = json.loads(actor_input_json) if actor_input_json else {}
    if not isinstance(actor_input, dict):
        raise ValueError("APIFY_ACTOR_INPUT_JSON must be a JSON object")
    actor_input["maxItems"] = max_items
    return json.dumps(actor_input)


def _resolve_ticker(ticker: str | None) -> str:
    value = (ticker if ticker is not None else input("Enter asset ticker: ")).strip().upper()
    if not value:
        raise SystemExit("Ticker cannot be empty.")
    return value


def main() -> None:
    load_environment()
    args = _parse_args()
    ticker = _resolve_ticker(args.ticker)
    actor_input_json = os.getenv("APIFY_ACTOR_INPUT_JSON")
    if actor_input_json:
        json.loads(actor_input_json)
    actor_input_json = _override_max_items(actor_input_json, args.max_items)

    price_actor_input_json = os.getenv("APIFY_PRICE_ACTOR_INPUT_JSON")
    if price_actor_input_json:
        json.loads(price_actor_input_json)

    try:
        counts = ingest(
            ticker=ticker,
            keywords=args.keyword,
            database_path=args.database,
            market_timezone=args.market_timezone,
            apify_actor_id=required_environment("APIFY_ACTOR_ID"),
            apify_price_actor_id=required_environment("APIFY_PRICE_ACTOR_ID"),
            apify_api_token=required_environment("APIFY_API_TOKEN"),
            tavily_api_key=required_environment("TAVILY_API_KEY"),
            actor_input_json=actor_input_json,
            price_actor_input_json=price_actor_input_json,
            lookback_days=args.days,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Ingestion stopped: {exc}") from None
    print(f"Ingested {counts[0]} X posts, {counts[1]} daily price bars, and {counts[2]} news results.")


if __name__ == "__main__":
    main()