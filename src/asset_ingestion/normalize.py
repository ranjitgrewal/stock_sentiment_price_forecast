from __future__ import annotations

import hashlib
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


def _first_value(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    normalized_record = {
        "".join(character.lower() for character in str(record_key) if character.isalnum()): value
        for record_key, value in record.items()
    }
    for key in keys:
        normalized_key = "".join(character.lower() for character in key if character.isalnum())
        value = normalized_record.get(normalized_key)
        if value is not None:
            return value
    return None


def _nested_value(record: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = record
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            return value
    return None


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_tweet(
    record: dict[str, Any],
    ticker: str,
    market_timezone: str,
    generate_missing_id: bool = False,
) -> dict[str, Any] | None:
    timestamp_value = _first_value(
        record,
        "createdAt",
        "created_at",
        "timestamp",
        "date",
        "datetime",
        "posted_at",
        "tweet_created_at",
        "post_date",
    )
    if timestamp_value is None:
        return None

    if isinstance(timestamp_value, (int, float)) or (
        isinstance(timestamp_value, str) and timestamp_value.strip().isdigit()
    ):
        timestamp = pd.to_datetime(float(timestamp_value), unit="s", utc=True, errors="coerce")
    else:
        timestamp = pd.to_datetime(timestamp_value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        return None

    author_followers = _first_value(
        record,
        "authorFollowers",
        "author_followers",
        "followersCount",
        "followers_count",
        "user_followers",
        "followers",
    )
    if author_followers is None:
        author_followers = _nested_value(
            record,
            ("author", "followers"),
            ("author", "followersCount"),
            ("user", "followersCount"),
            ("user", "followers_count"),
            ("user", "userFollowersCount"),
        )

    created_at_utc = timestamp.to_pydatetime().astimezone(timezone.utc)
    market_date = timestamp.tz_convert(market_timezone).strftime("%Y-%m-%d")
    tweet_id = _first_value(
        record, "id", "tweetId", "tweet_id", "post_id", "rest_id", "Tweet ID", "Tweet Id", "Post ID"
    )
    text = _first_value(record, "text", "fullText", "full_text", "content", "tweet", "tweet_text", "body")
    if text is None:
        return None
    if tweet_id is None:
        if not generate_missing_id:
            return None
        author = _first_value(record, "user_name", "username", "author", "user") or ""
        identity = f"{timestamp.isoformat()}\0{author}\0{text}"
        tweet_id = f"historical:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"

    return {
        "tweet_id": str(tweet_id),
        "ticker": ticker.upper(),
        "created_at_utc": created_at_utc.isoformat(),
        "market_date": market_date,
        "text": str(text),
        "likes": _integer(
            _first_value(record, "likeCount", "like_count", "likes", "favorite_count", "like_num")
        ),
        "reposts": _integer(
            _first_value(
                record,
                "retweetCount",
                "retweet_count",
                "repostCount",
                "retweets",
                "reposts",
                "retweet_num",
            )
        ),
        "views": _integer(_first_value(record, "viewCount", "view_count", "views", "impression_count")),
        "author_followers": _integer(author_followers),
        "raw_json": record,
    }


def normalize_price_bars(
    records: list[dict[str, Any]], ticker: str, market_timezone: str
) -> list[dict[str, Any]]:
    bars: list[dict[str, Any]] = []
    price_records: list[dict[str, Any]] = []
    for record in records:
        nested_history = _first_value(record, "priceHistory", "price_history", "history")
        if isinstance(nested_history, list):
            price_records.extend(item for item in nested_history if isinstance(item, dict))
        else:
            price_records.append(record)

    for record in price_records:
        timestamp_value = _first_value(record, "date", "timestamp", "datetime", "time")
        if timestamp_value is None:
            continue
        if isinstance(timestamp_value, str) and len(timestamp_value) == 10:
            try:
                timestamp = pd.Timestamp(timestamp_value).tz_localize(market_timezone)
            except (TypeError, ValueError):
                continue
        else:
            timestamp = pd.to_datetime(timestamp_value, errors="coerce")
            if pd.isna(timestamp):
                continue
            timestamp = pd.Timestamp(timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize(market_timezone)
        market_date = timestamp.tz_convert(market_timezone).strftime("%Y-%m-%d")
        bars.append(
            {
                "ticker": ticker.upper(),
                "market_date": market_date,
                "bar_at_utc": timestamp.tz_convert("UTC").isoformat(),
                "open": _number(_first_value(record, "open", "Open")),
                "high": _number(_first_value(record, "high", "High")),
                "low": _number(_first_value(record, "low", "Low")),
                "close": _number(_first_value(record, "close", "Close")),
                "adj_close": _number(_first_value(record, "adj_close", "adjClose", "Adj Close")),
                "volume": _integer(_first_value(record, "volume", "Volume")),
            }
        )
    return sorted(bars, key=lambda bar: bar["market_date"])


def _number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def normalize_news(record: dict[str, Any], ticker: str, market_timezone: str, retrieved_at: datetime) -> dict[str, Any]:
    published = record.get("published_date")
    if isinstance(published, str) and len(published) == 10:
        try:
            market_date = date.fromisoformat(published).isoformat()
            published_at_utc = None
        except ValueError:
            published = None
    if not published:
        market_date = retrieved_at.astimezone(ZoneInfo(market_timezone)).date().isoformat()
        published_at_utc = None
    elif not (isinstance(published, str) and len(published) == 10):
        published_timestamp = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(published_timestamp):
            market_date = retrieved_at.astimezone(ZoneInfo(market_timezone)).date().isoformat()
            published_at_utc = None
        else:
            market_date = published_timestamp.tz_convert(market_timezone).strftime("%Y-%m-%d")
            published_at_utc = published_timestamp.to_pydatetime().isoformat()

    return {
        "ticker": ticker.upper(),
        "url": record.get("url", ""),
        "title": record.get("title", ""),
        "content": record.get("content", ""),
        "published_at_utc": published_at_utc,
        "market_date": market_date,
        "relevance_score": _number(record.get("score")),
        "retrieved_at_utc": retrieved_at.astimezone(timezone.utc).isoformat(),
        "raw_json": record,
    }