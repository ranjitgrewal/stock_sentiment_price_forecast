import argparse
import json
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import pandas as pd

from asset_ingestion import providers
from asset_ingestion import cli
from asset_ingestion.import_history import import_history, load_records, load_tweet_ids_for_ticker
from asset_ingestion.cli import _override_max_items, _positive_int, _resolve_ticker
from asset_ingestion.database import connect_database, persist_ingestion
from asset_ingestion.features import (
    build_daily_features,
    persist_daily_features,
    score_text,
    write_sentiment_summary,
)
from asset_ingestion.model import prepare_model_data, walk_forward_validation
from asset_ingestion import model as model_module
from asset_ingestion import pipeline as pipeline_module
from asset_ingestion.visualize import generate_sentiment_price_chart
from asset_ingestion.normalize import normalize_news, normalize_price_bars, normalize_tweet


def test_normalize_tweet_converts_to_utc_and_market_date():
    tweet = normalize_tweet(
        {
            "id": "post-1",
            "createdAt": "2025-01-02T02:00:00-05:00",
            "text": "NVDA update",
            "likeCount": 4,
            "retweetCount": 2,
            "viewCount": 50,
            "author": {"followers": 1200},
        },
        "nvda",
        "America/New_York",
    )

    assert tweet is not None
    assert tweet["created_at_utc"] == "2025-01-02T07:00:00+00:00"
    assert tweet["market_date"] == "2025-01-02"
    assert tweet["author_followers"] == 1200
    assert tweet["reposts"] == 2


def test_import_history_reads_csv_headers_and_persists_tweets(tmp_path):
    csv_path = tmp_path / "historical_tweets.csv"
    csv_path.write_text(
        "tweet_id,writer,post_date,body,comment_num,retweet_num,like_num\n"
        "history-1,author,1420070457,NVDA historical post,2,3,12\n"
        "unrelated,author,1420070496,weather update,0,0,0\n",
        encoding="utf-8",
    )

    imported, filtered, invalid = import_history(
        str(csv_path), "NVDA", str(tmp_path / "history.sqlite3"), "America/New_York"
    )
    connection = connect_database(str(tmp_path / "history.sqlite3"))
    tweet = connection.execute("SELECT * FROM tweets WHERE tweet_id='history-1'").fetchone()
    connection.close()

    assert (imported, filtered, invalid) == (1, 1, 0)
    assert tweet["ticker"] == "NVDA"
    assert tweet["text"] == "NVDA historical post"
    assert tweet["likes"] == 12
    assert tweet["reposts"] == 3
    assert tweet["created_at_utc"] == "2015-01-01T00:00:57+00:00"
    assert tweet["views"] is None
    assert tweet["author_followers"] is None


def test_import_history_is_ticker_agnostic_and_matches_export_headers(tmp_path):
    csv_path = tmp_path / "historical_posts.csv"
    csv_path.write_text(
        "Post ID,Post Date,Tweet Text,Like Count,Retweet Count\n"
        "tsla-1,1735812000,Tesla TSLA update,8,2\n"
        "other-1,1735812001,Weather today,1,0\n",
        encoding="utf-8",
    )

    imported, filtered, invalid = import_history(
        str(csv_path), "tsla", str(tmp_path / "tsla.sqlite3"), "America/New_York", search_terms=["TSLA", "Tesla"]
    )

    connection = connect_database(str(tmp_path / "tsla.sqlite3"))
    tweet = connection.execute("SELECT * FROM tweets WHERE tweet_id='tsla-1'").fetchone()
    connection.close()

    assert (imported, filtered, invalid) == (1, 1, 0)
    assert tweet["ticker"] == "TSLA"
    assert tweet["text"] == "Tesla TSLA update"
    assert tweet["likes"] == 8
    assert tweet["reposts"] == 2


def test_import_history_matches_bitcoin_alias_for_btc_usd(tmp_path):
    csv_path = tmp_path / "bitcoin_posts.csv"
    csv_path.write_text(
        "id,timestamp,tweet,likes,retweets\n"
        "btc-1,1735812000,Bitcoin market outlook,8,2\n"
        "other-1,1735812001,Weather today,1,0\n",
        encoding="utf-8",
    )

    imported, filtered, invalid = import_history(
        str(csv_path), "BTC-USD", str(tmp_path / "bitcoin.sqlite3"), "America/New_York"
    )

    connection = connect_database(str(tmp_path / "bitcoin.sqlite3"))
    tweet = connection.execute("SELECT * FROM tweets WHERE tweet_id='btc-1'").fetchone()
    connection.close()

    assert (imported, filtered, invalid) == (1, 1, 0)
    assert tweet["ticker"] == "BTC-USD"
    assert tweet["text"] == "Bitcoin market outlook"


def test_import_history_generates_stable_id_for_bitcoin_archive_rows(tmp_path):
    csv_path = tmp_path / "bitcoin_archive.csv"
    csv_path.write_text(
        "user_name,date,text,user_followers,likes,retweets\n"
        'CryptoUser,2021-02-10 23:59:04,Bitcoin price update,42,3,1\n',
        encoding="utf-8",
    )
    database_path = str(tmp_path / "bitcoin_archive.sqlite3")

    first_result = import_history(
        str(csv_path), "BTC-USD", database_path, "UTC"
    )
    first_connection = connect_database(database_path)
    first_tweet = first_connection.execute("SELECT * FROM tweets").fetchone()
    first_connection.close()
    second_result = import_history(
        str(csv_path), "BTC-USD", database_path, "UTC"
    )
    second_connection = connect_database(database_path)
    tweet_count = second_connection.execute("SELECT COUNT(*) FROM tweets").fetchone()[0]
    second_connection.close()

    assert first_result == (1, 0, 0)
    assert second_result == (1, 0, 0)
    assert first_tweet["tweet_id"].startswith("historical:")
    assert tweet_count == 1
    assert first_tweet["market_date"] == "2021-02-10"


def test_import_history_uses_company_tweet_mapping(tmp_path):
    posts_path = tmp_path / "posts.csv"
    posts_path.write_text(
        "tweet_id,writer,post_date,body,comment_num,retweet_num,like_num\n"
        "amzn-1,author,1420070457,Cloud services update,1,2,5\n"
        "other-1,author,1420070496,Amazon mention but unmapped,0,0,1\n",
        encoding="utf-8",
    )
    mapping_path = tmp_path / "company_tweet.csv"
    mapping_path.write_text(
        "tweet_id,ticker_symbol\namzn-1,AMZN\nother-1,TSLA\n",
        encoding="utf-8",
    )

    assert load_tweet_ids_for_ticker(mapping_path, "amzn") == {"amzn-1"}
    imported, filtered, invalid = import_history(
        str(posts_path),
        "AMZN",
        str(tmp_path / "mapped.sqlite3"),
        "America/New_York",
        company_tweet_path=str(mapping_path),
    )

    connection = connect_database(str(tmp_path / "mapped.sqlite3"))
    tweet = connection.execute("SELECT * FROM tweets").fetchone()
    connection.close()

    assert (imported, filtered, invalid) == (1, 1, 0)
    assert tweet["ticker"] == "AMZN"
    assert tweet["text"] == "Cloud services update"


def test_load_records_reads_wrapped_json_list(tmp_path):
    json_path = tmp_path / "historical_tweets.json"
    json_path.write_text(
        '{"tweets":[{"id":"history-1","createdAt":"2025-01-02T15:00:00Z","text":"NVDA"}]}',
        encoding="utf-8",
    )

    assert load_records(json_path) == [
        {"id": "history-1", "createdAt": "2025-01-02T15:00:00Z", "text": "NVDA"}
    ]


def test_sentiment_lexicon_scores_and_negation():
    assert score_text("great growth and strong profits") > 0
    assert score_text("weak results and losses") < 0
    assert score_text("not good") < 0
    assert score_text("ordinary market update") == 0


def test_daily_features_include_zero_post_days_lags_and_horizons(tmp_path):
    connection = connect_database(str(tmp_path / "features.sqlite3"))
    bars = normalize_price_bars(
        [
            {
                "date": f"2025-01-{day:02d}",
                "open": 10.0 * day,
                "high": 10.0 * day + 1,
                "low": 10.0 * day - 1,
                "close": 10.0 * day,
                "volume": 100,
            }
            for day in range(1, 7)
        ],
        "TEST",
        "UTC",
    )
    tweet = normalize_tweet(
        {"id": "feature-1", "createdAt": "2025-01-02T12:00:00Z", "text": "strong growth"},
        "TEST",
        "UTC",
    )
    persist_ingestion(connection, "TEST", [tweet] if tweet else [], bars, [])

    features = build_daily_features(connection, "TEST")
    persist_daily_features(connection, features, "TEST")

    assert len(features) == 6
    assert features.loc[0, "tweet_count"] == 0
    assert features.loc[1, "tweet_count"] == 1
    assert features.loc[1, "sentiment_mean"] > 0
    assert features.loc[1, "sentiment_positive_share"] == 1
    assert pd.isna(features.loc[0, "return_1d_lag"])
    assert features.loc[2, "return_1d_lag"] == pytest.approx(1.0)
    assert features.loc[0, "target_close_n0"] == 10.0
    assert features.loc[0, "target_close_n5"] == 60.0
    saved = connection.execute("SELECT COUNT(*) FROM daily_features WHERE ticker='TEST'").fetchone()[0]
    connection.close()
    assert saved == 6


def test_sentiment_summary_reports_sparse_recent_post_coverage(tmp_path):
    features = pd.DataFrame(
        {
            "ticker": ["TEST"] * 4,
            "market_date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
            "close": [10.0, 11.0, 10.5, 12.0],
            "tweet_count": [0, 1, 0, 0],
            "sentiment_mean": [0.0, 0.2, 0.0, 0.0],
        }
    )
    summary_path = tmp_path / "summary.md"

    write_sentiment_summary(features, "TEST", summary_path)
    summary = summary_path.read_text(encoding="utf-8")

    assert "Insufficient X coverage" in summary
    assert "1 sessions with posts" in summary


def test_walk_forward_validation_builds_horizon_metrics():
    features = pd.DataFrame(
        {
            "ticker": ["TEST"] * 12,
            "market_date": [f"2025-01-{day:02d}" for day in range(1, 13)],
            "close": [10, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7],
            "tweet_count": [0, 1, 2, 2, 1, 0, 1, 2, 3, 2, 1, 0],
            "sentiment_mean": [0.0, 0.10, 0.20, 0.05, -0.10, 0.0, 0.15, 0.25, 0.05, -0.05, 0.0, 0.10],
            "sentiment_positive_share": [0.0, 0.5, 0.6, 0.4, 0.2, 0.0, 0.8, 0.7, 0.5, 0.3, 0.2, 0.6],
            "sentiment_negative_share": [0.0, 0.2, 0.1, 0.2, 0.3, 0.0, 0.1, 0.1, 0.2, 0.3, 0.2, 0.1],
            "sentiment_3d_mean": [0.0, 0.05, 0.10, 0.15, 0.10, 0.00, 0.05, 0.15, 0.20, 0.10, 0.00, 0.05],
            "sentiment_5d_mean": [0.0, 0.04, 0.08, 0.12, 0.10, 0.05, 0.08, 0.12, 0.16, 0.12, 0.06, 0.05],
            "return_1d_lag": [None, 0.05, 0.04, 0.05, 0.04, 0.04, 0.04, 0.03, 0.02, 0.04, 0.04, 0.02],
            "return_5d_lag": [None, None, 0.10, 0.09, 0.11, 0.09, 0.10, 0.09, 0.08, 0.10, 0.09, 0.08],
            "return_20d_lag": [None, None, None, None, None, None, None, None, None, None, None, None],
            "volatility_5d_lag": [None, None, 0.02, 0.03, 0.025, 0.02, 0.03, 0.025, 0.02, 0.025, 0.02, 0.015],
            "volatility_20d_lag": [None, None, None, None, None, None, None, None, None, None, None, None],
            "target_close_n0": [10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1],
            "target_close_n1": [11.0, 11.5, 12.0, 12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1, 15.4],
            "target_close_n2": [11.5, 12.0, 12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1, 15.4, 15.8],
            "target_close_n3": [12.0, 12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1, 15.4, 15.8, 16.2],
            "target_close_n4": [12.5, 13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1, 15.4, 15.8, 16.2, 16.6],
            "target_close_n5": [13.0, 13.4, 13.2, 13.8, 14.1, 14.7, 15.1, 15.4, 15.8, 16.2, 16.6, 16.9],
        }
    )

    X, y = prepare_model_data(features, horizon=0)
    results = walk_forward_validation(features, horizon=0, train_window=5, validation_window=2)

    assert X.shape[0] > 0
    assert y.shape[0] == X.shape[0]
    assert "mae" in results
    assert "rmse" in results
    assert "direction_accuracy" in results
    assert results["n_observations"] > 0
    assert results["n_folds"] >= 1

    forest_results = walk_forward_validation(
        features,
        horizon=0,
        train_window=5,
        validation_window=2,
        model_type="random_forest",
    )
    assert forest_results["model"] == "random_forest"
    assert forest_results["n_observations"] == results["n_observations"]


def test_build_horizon_summary_compares_models_for_each_horizon(monkeypatch):
    calls = []

    def fake_validation(features, horizon, **kwargs):
        calls.append((horizon, kwargs["model_type"], kwargs["validation_window"]))
        return {
            "target_column": f"target_close_n{horizon}",
            "mae": 1.0,
            "rmse": 1.0,
            "direction_accuracy": 0.5,
            "n_observations": 10,
            "n_folds": 2,
        }

    monkeypatch.setattr(model_module, "walk_forward_validation", fake_validation)
    summary = model_module.build_horizon_summary(pd.DataFrame({"market_date": range(60)}))

    assert len(summary) == 12
    assert set(summary["model"]) == {"linear", "random_forest"}
    assert {horizon for horizon, _, _ in calls} == set(range(6))
    assert {model_type for _, model_type, _ in calls} == {"linear", "random_forest"}
    assert {validation_window for _, _, validation_window in calls} == {5}


def test_generate_sentiment_price_chart_writes_png(tmp_path):
    features = pd.DataFrame(
        {
            "ticker": ["TEST"] * 6,
            "market_date": [f"2025-01-{day:02d}" for day in range(1, 7)],
            "close": [10.0, 10.5, 11.0, 11.3, 11.8, 12.2],
            "tweet_count": [0, 1, 2, 3, 1, 0],
            "sentiment_mean": [0.0, 0.1, 0.2, 0.05, -0.02, 0.0],
            "sentiment_3d_mean": [0.0, 0.05, 0.1, 0.12, 0.08, 0.02],
        }
    )
    output_path = tmp_path / "sentiment_vs_price.png"

    result = generate_sentiment_price_chart(features, output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_model_command_writes_report_and_chart_together(tmp_path, monkeypatch):
    database_path = tmp_path / "model_outputs.sqlite3"
    connection = connect_database(str(database_path))
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    bars = [
        {
            "ticker": "TEST",
            "market_date": timestamp.strftime("%Y-%m-%d"),
            "bar_at_utc": timestamp.isoformat(),
            "open": 10.0 + index,
            "high": 11.0 + index,
            "low": 9.0 + index,
            "close": 10.5 + index,
            "adj_close": 10.5 + index,
            "volume": 100,
        }
        for index, timestamp in enumerate(dates)
    ]
    persist_ingestion(connection, "TEST", [], bars, [])
    features = build_daily_features(connection, "TEST")
    persist_daily_features(connection, features, "TEST")
    connection.close()

    report_path = tmp_path / "reports" / "test_forecast.md"
    monkeypatch.setattr(
        model_module,
        "_parse_args",
        lambda: SimpleNamespace(
            ticker="TEST",
            database=str(database_path),
            output=str(report_path),
            chart_output=None,
            train_window=None,
        ),
    )
    monkeypatch.setattr(
        model_module,
        "build_horizon_summary",
        lambda features, train_window: pd.DataFrame(
            [
                {
                    "model": model_name,
                    "horizon": 1,
                    "target_column": "target_close_n1",
                    "mae": mae,
                    "rmse": mae + 0.5,
                    "direction_accuracy": 0.5,
                    "n_observations": 10,
                    "n_folds": 2,
                }
                for model_name, mae in (("linear", 1.0), ("random_forest", 0.8))
            ]
        ),
    )

    model_module.main()

    assert report_path.exists()
    assert (report_path.parent / "test_sentiment_vs_price.png").exists()
    report = report_path.read_text(encoding="utf-8")
    assert "| Horizon | Linear MAE | Random Forest MAE |" in report
    assert "Across n+1 to n+5" in report


def test_pipeline_generates_reports_and_chart_from_csv(tmp_path, monkeypatch):
    dates = pd.date_range("2026-01-01", periods=35, freq="D")
    csv_path = tmp_path / "posts.csv"
    csv_rows = ["id,date,text"]
    for index, timestamp in enumerate(dates):
        csv_rows.append(
            f"post-{index},{timestamp.strftime('%Y-%m-%d')}T12:00:00Z,TEST bullish growth {index}"
        )
    csv_path.write_text("\n".join(csv_rows), encoding="utf-8")

    def fake_live_ingestion(ticker, keywords, database_path, market_timezone, max_items, lookback_days):
        connection = connect_database(database_path)
        bars = [
            {
                "ticker": ticker,
                "market_date": timestamp.strftime("%Y-%m-%d"),
                "bar_at_utc": timestamp.isoformat(),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "adj_close": 100.5 + index,
                "volume": 1000,
            }
            for index, timestamp in enumerate(dates)
        ]
        persist_ingestion(connection, ticker, [], bars, [])
        connection.close()
        return 0, len(bars), 0

    monkeypatch.setattr(pipeline_module, "load_environment", lambda: None)
    monkeypatch.setattr(pipeline_module, "_run_recent_ingestion", fake_live_ingestion)

    result = pipeline_module.run_pipeline(
        ticker="test",
        data_file=csv_path,
        database_path=tmp_path / "pipeline.sqlite3",
        market_timezone="UTC",
        keywords=[],
        output_dir=tmp_path / "reports",
    )

    assert result["ticker"] == "TEST"
    assert result["history_counts"][0] == 35
    assert result["feature_rows"] == 35
    assert result["overlap_days"] == 35
    assert result["sentiment_report"].exists()
    assert result["forecast_report"].exists()
    assert result["chart"].exists()
    assert result["overlap_decision"] == "matched_existing_data"
    assert "35 distinct dates" in result["sentiment_report"].read_text(encoding="utf-8")
    assert "35 distinct dates" in result["forecast_report"].read_text(encoding="utf-8")


def test_pipeline_runs_from_recent_scrape_without_historical_csv(tmp_path, monkeypatch):
    dates = pd.date_range("2026-09-01", periods=35, freq="D")
    database_path = tmp_path / "recent_only.sqlite3"

    def fake_live_ingestion(ticker, keywords, database, market_timezone, max_items, lookback_days):
        assert lookback_days == 60
        connection = connect_database(database)
        tweets = [
            normalize_tweet(
                {
                    "id": f"recent-{index}",
                    "createdAt": timestamp.isoformat(),
                    "text": "TEST bullish growth",
                },
                ticker,
                market_timezone,
            )
            for index, timestamp in enumerate(dates)
        ]
        bars = [
            {
                "ticker": ticker,
                "market_date": timestamp.strftime("%Y-%m-%d"),
                "bar_at_utc": timestamp.isoformat(),
                "open": 100.0 + index,
                "high": 101.0 + index,
                "low": 99.0 + index,
                "close": 100.5 + index,
                "adj_close": 100.5 + index,
                "volume": 1000,
            }
            for index, timestamp in enumerate(dates)
        ]
        persist_ingestion(connection, ticker, [tweet for tweet in tweets if tweet], bars, [])
        connection.close()
        return len(tweets), len(bars), 0

    monkeypatch.setattr(pipeline_module, "load_environment", lambda: None)
    monkeypatch.setattr(pipeline_module, "_run_recent_ingestion", fake_live_ingestion)

    result = pipeline_module.run_pipeline(
        ticker="TEST",
        data_file=None,
        database_path=database_path,
        market_timezone="UTC",
        keywords=[],
        output_dir=tmp_path / "recent_reports",
        lookback_days=60,
    )

    assert result["live_counts"] == (35, 35, 0)
    assert result["history_counts"] == (0, 0, 0)
    assert result["overlap_days"] == 35
    assert result["forecast_report"].exists()
    assert result["chart"].exists()


def test_ingest_uses_requested_lookback_for_all_sources(tmp_path, monkeypatch):
    start_dates = {}

    def fake_x_posts(*args):
        start_dates["x"] = args[2]
        return []

    def fake_prices(*args):
        start_dates["price"] = args[1]
        return []

    def fake_news(ticker, api_key, start_date, max_results=20):
        start_dates["news"] = start_date
        return []

    monkeypatch.setattr(cli, "fetch_x_posts", fake_x_posts)
    monkeypatch.setattr(cli, "fetch_price_history", fake_prices)
    monkeypatch.setattr(cli, "fetch_news", fake_news)

    cli.ingest(
        ticker="TEST",
        keywords=[],
        database_path=str(tmp_path / "lookback.sqlite3"),
        market_timezone="UTC",
        apify_actor_id="test-x",
        apify_price_actor_id="test-price",
        apify_api_token="test-token",
        tavily_api_key="test-key",
        end_date=date(2025, 1, 3),
        lookback_days=60,
    )

    expected_start = date(2025, 1, 3) - timedelta(days=60)
    assert start_dates == {"x": expected_start, "price": expected_start, "news": expected_start}


def test_zero_overlap_prompt_imports_alternative_csv(tmp_path, monkeypatch):
    database_path = str(tmp_path / "overlap.sqlite3")
    connection = connect_database(database_path)
    persist_ingestion(
        connection,
        "BTC-USD",
        [],
        [
            {
                "ticker": "BTC-USD",
                "market_date": "2026-01-02",
                "bar_at_utc": "2026-01-02T00:00:00+00:00",
                "open": 100.0,
                "high": 102.0,
                "low": 99.0,
                "close": 101.0,
                "adj_close": 101.0,
                "volume": 100,
            }
        ],
        [],
    )
    connection.close()
    alternate_csv = tmp_path / "recent_posts.csv"
    alternate_csv.write_text(
        "id,date,text\nrecent-1,2026-01-02T12:00:00Z,Bitcoin BTC market update\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("builtins.input", lambda _: str(alternate_csv))

    counts, coverage, decision, additional_files = pipeline_module._resolve_zero_overlap(
        "BTC-USD", database_path, "UTC", [], (0, 0, 0), prompt_user=True
    )

    assert counts == (1, 0, 0)
    assert coverage["overlap_days"] == 1
    assert decision == "alternate_file_added_overlap"
    assert additional_files == [str(alternate_csv.resolve())]


def test_zero_overlap_prompt_can_proceed_with_current_data(tmp_path, monkeypatch):
    database_path = str(tmp_path / "no_overlap.sqlite3")
    connection = connect_database(database_path)
    connection.close()
    monkeypatch.setattr("builtins.input", lambda _: "P")

    counts, coverage, decision, additional_files = pipeline_module._resolve_zero_overlap(
        "BTC-USD", database_path, "UTC", [], (3, 0, 0), prompt_user=True
    )

    assert counts == (3, 0, 0)
    assert coverage["overlap_days"] == 0
    assert decision == "proceeded_without_overlap"
    assert additional_files == []


def test_zero_overlap_prompt_can_quit(tmp_path, monkeypatch):
    database_path = str(tmp_path / "quit.sqlite3")
    connection = connect_database(database_path)
    connection.close()
    monkeypatch.setattr("builtins.input", lambda _: "Q")

    with pytest.raises(ValueError, match="Pipeline cancelled"):
        pipeline_module._resolve_zero_overlap(
            "BTC-USD", database_path, "UTC", [], (0, 0, 0), prompt_user=True
        )


def test_pipeline_report_note_explains_proceeding_without_overlap():
    note = pipeline_module._data_quality_note(
        {
            "tweet_start": "2014-12-31",
            "tweet_end": "2019-12-31",
            "price_start": "2026-05-04",
            "price_end": "2026-10-29",
            "overlap_days": 0,
        },
        "proceeded_without_overlap",
        [],
    )

    assert "proceeded with insufficient date overlap" in note
    assert "2014-12-31 to 2019-12-31" in note
    assert "2026-05-04 to 2026-10-29" in note
    assert "price-derived inputs only" in note
    assert "not observed sentiment" in note


def test_pipeline_report_note_records_alternate_file_and_rebuild():
    note = pipeline_module._data_quality_note(
        {
            "tweet_start": "2026-01-02",
            "tweet_end": "2026-01-02",
            "price_start": "2026-01-02",
            "price_end": "2026-01-02",
            "overlap_days": 1,
        },
        "alternate_file_added_overlap",
        ["/tmp/recent_aapl.csv"],
    )

    assert "1 distinct dates" in note
    assert "/tmp/recent_aapl.csv" in note
    assert "imported before downstream features and reports were regenerated" in note


def test_fetch_price_history_uses_configured_apify_actor(monkeypatch):
    expected_records = [{"date": "2025-01-02", "open": 10, "close": 11}]

    class FakeDataset:
        def iterate_items(self):
            return iter(expected_records)

    class FakeActor:
        def call(self, run_input):
            assert run_input == {
                "ticker": "NVDA",
                "startDate": "2025-01-01",
                "endDate": "2025-01-03",
            }
            return {"defaultDatasetId": "price-dataset"}

    class FakeClient:
        def __init__(self, token):
            assert token == "test-token"

        def actor(self, actor_id):
            assert actor_id == "user~price-actor"
            return FakeActor()

        def dataset(self, dataset_id):
            assert dataset_id == "price-dataset"
            return FakeDataset()

    monkeypatch.setattr(providers, "ApifyClient", FakeClient)
    records = providers.fetch_price_history(
        "nvda",
        datetime(2025, 1, 1).date(),
        datetime(2025, 1, 3).date(),
        "user~price-actor",
        "test-token",
        '{"ticker":"$TICKER","startDate":"$START_DATE","endDate":"$END_DATE"}',
    )

    assert records == expected_records


def test_fetch_x_posts_accepts_typed_apify_run(monkeypatch):
    expected_records = [{"id": "post-1", "text": "NVDA update"}]

    class FakeDataset:
        def iterate_items(self):
            return iter(expected_records)

    class FakeActor:
        def call(self, run_input):
            assert run_input["searchTerms"] == [
                '"NVDA" since:2025-01-01 until:2025-01-03',
                '"$NVDA" since:2025-01-01 until:2025-01-03',
            ]
            return SimpleNamespace(default_dataset_id="tweet-dataset")

    class FakeClient:
        def __init__(self, token):
            assert token == "test-token"

        def actor(self, actor_id):
            assert actor_id == "apidojo/tweet-scraper"
            return FakeActor()

        def dataset(self, dataset_id):
            assert dataset_id == "tweet-dataset"
            return FakeDataset()

    monkeypatch.setattr(providers, "ApifyClient", FakeClient)
    records = providers.fetch_x_posts(
        "NVDA",
        [],
        datetime(2025, 1, 1).date(),
        datetime(2025, 1, 3).date(),
        "apidojo/tweet-scraper",
        "test-token",
        '{"searchTerms":["$CASHTAG $DATE_FILTER"],"maxItems":20}',
    )

    assert records == expected_records


def test_fetch_x_posts_searches_each_keyword_separately(monkeypatch):
    expected_records = [{"id": "post-1", "text": "NVDA update"}]
    expected_terms = [
        '"NVDA" since:2025-01-01 until:2025-01-03',
        '"$NVDA" since:2025-01-01 until:2025-01-03',
        '"NVIDIA" since:2025-01-01 until:2025-01-03',
        '"AI chips" since:2025-01-01 until:2025-01-03',
    ]

    class FakeDataset:
        def iterate_items(self):
            return iter(expected_records)

    class FakeActor:
        def call(self, run_input):
            assert run_input == {"searchTerms": expected_terms, "maxItems": 100}
            return SimpleNamespace(default_dataset_id="tweet-dataset")

    class FakeClient:
        def __init__(self, token):
            assert token == "test-token"

        def actor(self, actor_id):
            assert actor_id == "apidojo/tweet-scraper"
            return FakeActor()

        def dataset(self, dataset_id):
            assert dataset_id == "tweet-dataset"
            return FakeDataset()

    monkeypatch.setattr(providers, "ApifyClient", FakeClient)
    records = providers.fetch_x_posts(
        "NVDA",
        ["NVIDIA", "AI chips"],
        datetime(2025, 1, 1).date(),
        datetime(2025, 1, 3).date(),
        "apidojo/tweet-scraper",
        "test-token",
        '{"searchTerms":"$SEARCH_TERMS","maxItems":100}',
    )

    assert records == expected_records


def test_fetch_x_posts_stops_on_empty_dataset(monkeypatch):
    class FakeDataset:
        def iterate_items(self):
            return iter(())

    class FakeActor:
        def call(self, run_input):
            return SimpleNamespace(default_dataset_id="empty-dataset")

    class FakeClient:
        def __init__(self, token):
            pass

        def actor(self, actor_id):
            return FakeActor()

        def dataset(self, dataset_id):
            return FakeDataset()

    monkeypatch.setattr(providers, "ApifyClient", FakeClient)
    with pytest.raises(RuntimeError, match="no posts.*monthly limits"):
        providers.fetch_x_posts(
            "NVDA",
            ["NVIDIA"],
            datetime(2025, 1, 1).date(),
            datetime(2025, 1, 3).date(),
            "apidojo/tweet-scraper",
            "test-token",
        )


def test_resolve_ticker_prompts_and_normalizes_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "  nvda  ")

    assert _resolve_ticker(None) == "NVDA"
    assert _resolve_ticker(" msft ") == "MSFT"


def test_resolve_ticker_rejects_blank_input(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: "   ")

    try:
        _resolve_ticker(None)
    except SystemExit as exc:
        assert str(exc) == "Ticker cannot be empty."
    else:
        raise AssertionError("Blank ticker should be rejected")


def test_max_items_override_changes_only_requested_run_cap():
    configured = '{"searchTerms":"$SEARCH_TERMS","maxItems":100}'

    overridden = _override_max_items(configured, 500)
    unchanged = _override_max_items(configured, None)

    assert json.loads(overridden)["maxItems"] == 500
    assert json.loads(unchanged)["maxItems"] == 100
    assert _positive_int("500") == 500


def test_max_items_override_rejects_zero():
    try:
        _positive_int("0")
    except argparse.ArgumentTypeError as exc:
        assert "positive integer" in str(exc)
    else:
        raise AssertionError("Zero max-items should be rejected")


def test_ingest_persists_and_aligns_x_price_and_news(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli,
        "fetch_x_posts",
        lambda *args: [
            {
                "id": "post-1",
                "createdAt": "2025-01-02T15:00:00Z",
                "text": "NVDA update",
                "likeCount": 4,
            }
        ],
    )
    monkeypatch.setattr(
        cli,
        "fetch_price_history",
        lambda *args: [
            {
                "symbol": "NVDA",
                "priceHistory": [
                    {
                        "date": "2025-01-02",
                        "open": 10,
                        "high": 12,
                        "low": 9,
                        "close": 11,
                        "volume": 100,
                    }
                ],
            }
        ],
    )
    monkeypatch.setattr(
        cli,
        "fetch_news",
        lambda *args: [
            {
                "url": "https://example.com/nvda",
                "title": "NVDA market update",
                "content": "Company update",
                "published_date": "2025-01-02",
                "score": 0.91,
            }
        ],
    )

    counts = cli.ingest(
        ticker="NVDA",
        keywords=["NVIDIA"],
        database_path=str(tmp_path / "x-only.sqlite3"),
        market_timezone="America/New_York",
        apify_actor_id="apidojo/tweet-scraper",
        apify_price_actor_id="pink_comic/yahoo-finance-stock-data",
        apify_api_token="test-token",
        tavily_api_key="test-tavily-key",
        price_actor_input_json='{"symbols":["$TICKER"],"dateRange":"6mo"}',
        end_date=datetime(2025, 1, 3).date(),
    )

    connection = connect_database(str(tmp_path / "x-only.sqlite3"))
    tweet = connection.execute("SELECT * FROM tweets").fetchone()
    price = connection.execute("SELECT * FROM price_bars").fetchone()
    news = connection.execute("SELECT * FROM news").fetchone()
    aligned = connection.execute("SELECT * FROM daily_alignment").fetchone()
    connection.close()

    assert counts == (1, 1, 1)
    assert tweet["ticker"] == "NVDA"
    assert tweet["text"] == "NVDA update"
    assert price["close"] == 11
    assert news["title"] == "NVDA market update"
    assert aligned["market_date"] == "2025-01-02"
    assert aligned["close"] == 11
    assert aligned["tweet_count"] == 1
    assert aligned["news_count"] == 1


def test_daily_alignment_joins_tweets_and_news_to_price_date(tmp_path):
    connection = connect_database(str(tmp_path / "test.sqlite3"))
    tweets = [
        normalize_tweet(
            {
                "id": "post-1",
                "createdAt": "2025-01-02T10:00:00-05:00",
                "text": "NVDA update",
                "likeCount": 4,
                "retweetCount": 2,
                "viewCount": 50,
                "author": {"followers": 1200},
            },
            "NVDA",
            "America/New_York",
        ),
        normalize_tweet(
            {
                "id": "post-2",
                "createdAt": "2025-01-02T11:00:00-05:00",
                "text": "More NVDA news",
                "likeCount": 3,
                "retweetCount": 1,
                "viewCount": 25,
                "author": {"followers": 800},
            },
            "NVDA",
            "America/New_York",
        ),
    ]
    bars = normalize_price_bars(
        [
            {
                "date": "2025-01-02",
                "open": 10,
                "high": 12,
                "low": 9,
                "close": 11,
                "adj_close": 11,
                "volume": 100,
            }
        ],
        "NVDA",
        "America/New_York",
    )
    news = [
        normalize_news(
            {"url": "https://example.com/news", "title": "NVDA update", "published_date": "2025-01-02"},
            "NVDA",
            "America/New_York",
            datetime(2025, 1, 2, 15, tzinfo=timezone.utc),
        ),
        normalize_news(
            {"url": "https://example.com/second", "title": "Second NVDA update", "published_date": "2025-01-02"},
            "NVDA",
            "America/New_York",
            datetime(2025, 1, 2, 15, tzinfo=timezone.utc),
        ),
    ]

    persist_ingestion(connection, "NVDA", [tweet for tweet in tweets if tweet], bars, news)
    row = connection.execute("SELECT * FROM daily_alignment").fetchone()
    connection.close()

    assert row["market_date"] == "2025-01-02"
    assert row["close"] == 11
    assert row["tweet_count"] == 2
    assert row["tweet_likes"] == 7
    assert row["tweet_reposts"] == 3
    assert row["news_count"] == 2