from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS tweets (
    tweet_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    market_date TEXT NOT NULL,
    text TEXT NOT NULL,
    likes INTEGER,
    reposts INTEGER,
    views INTEGER,
    author_followers INTEGER,
    raw_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tweets_ticker_date ON tweets(ticker, market_date);

CREATE TABLE IF NOT EXISTS price_bars (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    bar_at_utc TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    PRIMARY KEY (ticker, market_date)
);

CREATE TABLE IF NOT EXISTS news (
    ticker TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    published_at_utc TEXT,
    market_date TEXT NOT NULL,
    relevance_score REAL,
    retrieved_at_utc TEXT NOT NULL,
    raw_json TEXT NOT NULL,
    PRIMARY KEY (ticker, url)
);
CREATE INDEX IF NOT EXISTS idx_news_ticker_date ON news(ticker, market_date);

CREATE TABLE IF NOT EXISTS daily_alignment (
    ticker TEXT NOT NULL,
    market_date TEXT NOT NULL,
    bar_at_utc TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adj_close REAL,
    volume INTEGER,
    tweet_count INTEGER NOT NULL,
    tweet_likes INTEGER NOT NULL,
    tweet_reposts INTEGER NOT NULL,
    tweet_views INTEGER NOT NULL,
    author_followers INTEGER NOT NULL,
    news_count INTEGER NOT NULL,
    news_titles_json TEXT NOT NULL,
    PRIMARY KEY (ticker, market_date)
);
"""


def connect_database(path: str) -> sqlite3.Connection:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def persist_tweets(connection: sqlite3.Connection, tweets: list[dict[str, Any]]) -> None:
    connection.executemany(
        """INSERT INTO tweets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(tweet_id) DO UPDATE SET
            ticker=excluded.ticker, created_at_utc=excluded.created_at_utc,
            market_date=excluded.market_date, text=excluded.text,
            likes=excluded.likes, reposts=excluded.reposts, views=excluded.views,
            author_followers=excluded.author_followers, raw_json=excluded.raw_json""",
        [
            (
                item["tweet_id"], item["ticker"], item["created_at_utc"], item["market_date"],
                item["text"], item["likes"], item["reposts"], item["views"],
                item["author_followers"], json.dumps(item["raw_json"], ensure_ascii=True),
            )
            for item in tweets
        ],
    )


def persist_ingestion(
    connection: sqlite3.Connection,
    ticker: str,
    tweets: list[dict[str, Any]],
    bars: list[dict[str, Any]],
    news_items: list[dict[str, Any]],
) -> None:
    with connection:
        persist_tweets(connection, tweets)
        connection.executemany(
            """INSERT INTO price_bars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, market_date) DO UPDATE SET
                bar_at_utc=excluded.bar_at_utc, open=excluded.open, high=excluded.high,
                low=excluded.low, close=excluded.close, adj_close=excluded.adj_close,
                volume=excluded.volume""",
            [
                (
                    item["ticker"], item["market_date"], item["bar_at_utc"], item["open"],
                    item["high"], item["low"], item["close"], item["adj_close"], item["volume"],
                )
                for item in bars
            ],
        )
        connection.executemany(
            """INSERT INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, url) DO UPDATE SET
                title=excluded.title, content=excluded.content,
                published_at_utc=excluded.published_at_utc, market_date=excluded.market_date,
                relevance_score=excluded.relevance_score, retrieved_at_utc=excluded.retrieved_at_utc,
                raw_json=excluded.raw_json""",
            [
                (
                    item["ticker"], item["url"], item["title"], item["content"],
                    item["published_at_utc"], item["market_date"], item["relevance_score"],
                    item["retrieved_at_utc"], json.dumps(item["raw_json"], ensure_ascii=True),
                )
                for item in news_items
            ],
        )
        connection.execute("DELETE FROM daily_alignment WHERE ticker = ?", (ticker.upper(),))
        connection.execute(
            """INSERT INTO daily_alignment (
                ticker, market_date, bar_at_utc, open, high, low, close, adj_close, volume,
                tweet_count, tweet_likes, tweet_reposts, tweet_views, author_followers,
                news_count, news_titles_json
            )
            SELECT p.ticker, p.market_date, p.bar_at_utc, p.open, p.high, p.low, p.close,
                p.adj_close, p.volume,
                COALESCE(t.tweet_count, 0), COALESCE(t.tweet_likes, 0),
                COALESCE(t.tweet_reposts, 0), COALESCE(t.tweet_views, 0),
                COALESCE(t.author_followers, 0), COALESCE(n.news_count, 0),
                COALESCE(n.news_titles_json, '[]')
            FROM price_bars p
            LEFT JOIN (
                SELECT ticker, market_date, COUNT(*) AS tweet_count,
                    SUM(COALESCE(likes, 0)) AS tweet_likes,
                    SUM(COALESCE(reposts, 0)) AS tweet_reposts,
                    SUM(COALESCE(views, 0)) AS tweet_views,
                    SUM(COALESCE(author_followers, 0)) AS author_followers
                FROM tweets GROUP BY ticker, market_date
            ) t ON t.ticker = p.ticker AND t.market_date = p.market_date
            LEFT JOIN (
                SELECT ticker, market_date, COUNT(*) AS news_count,
                    json_group_array(title) AS news_titles_json
                FROM news GROUP BY ticker, market_date
            ) n ON n.ticker = p.ticker AND n.market_date = p.market_date
            WHERE p.ticker = ?
            """,
            (ticker.upper(),),
        )