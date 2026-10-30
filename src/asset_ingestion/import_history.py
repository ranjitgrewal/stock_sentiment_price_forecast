from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterator

from asset_ingestion.database import connect_database, persist_ingestion, persist_tweets
from asset_ingestion.normalize import normalize_tweet
from asset_ingestion.providers import load_environment


_RECORD_KEYS = ("tweets", "posts", "items", "results", "data")


def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as input_file:
            yield from csv.DictReader(input_file)
        return

    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Every historical tweet record must be a JSON object")
                    yield record
        return

    if suffix == ".json":
        with source.open("r", encoding="utf-8") as input_file:
            payload = json.load(input_file)
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict):
            records = next((payload[key] for key in _RECORD_KEYS if isinstance(payload.get(key), list)), None)
            if records is None:
                records = [payload]
        else:
            raise ValueError("JSON input must be a tweet object, a list, or an object containing a tweet list")
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("Every historical tweet record must be a JSON object")
        yield from records
        return

    raise ValueError("Historical input must be .csv, .json, or .jsonl")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_records(path))


def _matches_search_terms(record: dict[str, Any], search_terms: list[str]) -> bool:
    normalized_record = {
        "".join(character.lower() for character in str(key) if character.isalnum()): value
        for key, value in record.items()
    }
    text_value = next(
        (normalized_record[key] for key in ("body", "text", "content", "tweet", "tweettext") if key in normalized_record),
        "",
    )
    text = str(text_value).casefold()
    return any(term.casefold() in text for term in search_terms if term)


def load_tweet_ids_for_ticker(mapping_path: str | Path, ticker: str) -> set[str]:
    tweet_ids: set[str] = set()
    with Path(mapping_path).open("r", encoding="utf-8-sig", newline="") as mapping_file:
        reader = csv.DictReader(mapping_file)
        if not reader.fieldnames or not {"tweet_id", "ticker_symbol"}.issubset(reader.fieldnames):
            raise ValueError("Company mapping CSV must contain tweet_id and ticker_symbol columns")
        for row in reader:
            if (row.get("ticker_symbol") or "").strip().casefold() == ticker.casefold():
                tweet_id = (row.get("tweet_id") or "").strip()
                if tweet_id:
                    tweet_ids.add(tweet_id)
    return tweet_ids


def import_history(
    path: str,
    ticker: str,
    database_path: str,
    market_timezone: str,
    search_terms: list[str] | None = None,
    batch_size: int = 5000,
    company_tweet_path: str | None = None,
) -> tuple[int, int, int]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    ticker = ticker.upper()
    terms = list(search_terms) if search_terms is not None else [ticker, f"${ticker}"]
    if ticker.split("-", maxsplit=1)[0] in {"BTC", "XBT"}:
        terms.extend(("BTC", "$BTC", "Bitcoin", "#Bitcoin"))
    terms = list(dict.fromkeys(terms))
    mapped_tweet_ids = (
        load_tweet_ids_for_ticker(company_tweet_path, ticker) if company_tweet_path else None
    )
    imported = 0
    filtered = 0
    invalid = 0
    batch: list[dict[str, Any]] = []
    connection = connect_database(database_path)
    try:
        for record in iter_records(path):
            if mapped_tweet_ids is not None:
                record_id = record.get("tweet_id", record.get("id", record.get("post_id")))
                if record_id is None or str(record_id).strip() not in mapped_tweet_ids:
                    filtered += 1
                    continue
            elif not _matches_search_terms(record, terms):
                filtered += 1
                continue
            tweet = normalize_tweet(record, ticker, market_timezone, generate_missing_id=True)
            if tweet is None:
                invalid += 1
                continue
            batch.append(tweet)
            if len(batch) >= batch_size:
                with connection:
                    persist_tweets(connection, batch)
                imported += len(batch)
                batch.clear()

        if batch:
            with connection:
                persist_tweets(connection, batch)
            imported += len(batch)

        with connection:
            persist_ingestion(connection, ticker, [], [], [])
    finally:
        connection.close()
    return imported, filtered, invalid


def main() -> None:
    load_environment()
    parser = argparse.ArgumentParser(description="Import historical X/Twitter posts from a local CSV or JSON export.")
    parser.add_argument("--ticker", required=True, help="Asset ticker used to tag imported posts")
    parser.add_argument("--file", required=True, help="Path to a CSV, JSON, or JSONL tweet export")
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Additional case-insensitive body term for selecting relevant posts; repeat as needed",
    )
    parser.add_argument(
        "--company-tweet-file",
        help="Optional CSV mapping with tweet_id,ticker_symbol; when supplied, selects exact ticker-linked posts",
    )
    parser.add_argument("--database", default=os.getenv("DATABASE_PATH", "data/market_data.sqlite3"))
    parser.add_argument("--market-timezone", default=os.getenv("MARKET_TIMEZONE", "America/New_York"))
    args = parser.parse_args()

    imported, filtered, invalid = import_history(
        args.file,
        args.ticker,
        args.database,
        args.market_timezone,
        search_terms=[args.ticker, f"${args.ticker}", *args.keyword],
        company_tweet_path=args.company_tweet_file,
    )
    print(f"Imported or updated {imported} posts; filtered {filtered} unrelated records; skipped {invalid} invalid records.")


if __name__ == "__main__":
    main()