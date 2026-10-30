from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from asset_ingestion.cli import ingest
from asset_ingestion.database import connect_database
from asset_ingestion.features import build_daily_features, persist_daily_features, write_sentiment_summary
from asset_ingestion.import_history import import_history
from asset_ingestion.model import build_horizon_summary, write_model_summary
from asset_ingestion.evaluation import write_risk_evaluation_report
from asset_ingestion.providers import load_environment, required_environment
from asset_ingestion.visualize import generate_sentiment_price_chart

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "src" / "data" / "market_data.sqlite3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "src" / "reports"


def _default_database_path() -> str:
    configured = os.getenv("DATABASE_PATH")
    if not configured:
        return str(DEFAULT_DATABASE_PATH)
    path = Path(configured).expanduser()
    if path.is_absolute():
        return str(path)
    source_relative = PROJECT_ROOT / "src" / path
    if source_relative.exists():
        return str(source_relative)
    return str(PROJECT_ROOT / path)


def _run_recent_ingestion(
    ticker: str,
    keywords: list[str],
    database_path: str,
    market_timezone: str,
    max_items: int | None,
    lookback_days: int,
) -> tuple[int, int, int]:
    actor_input_json = os.getenv("APIFY_ACTOR_INPUT_JSON")
    actor_input = json.loads(actor_input_json) if actor_input_json else {}
    if not isinstance(actor_input, dict):
        raise ValueError("APIFY_ACTOR_INPUT_JSON must be a JSON object")
    if max_items is not None:
        actor_input["maxItems"] = max_items

    price_actor_input_json = os.getenv("APIFY_PRICE_ACTOR_INPUT_JSON")
    if price_actor_input_json:
        json.loads(price_actor_input_json)

    return ingest(
        ticker=ticker,
        keywords=keywords,
        database_path=database_path,
        market_timezone=market_timezone,
        apify_actor_id=required_environment("APIFY_ACTOR_ID"),
        apify_price_actor_id=required_environment("APIFY_PRICE_ACTOR_ID"),
        apify_api_token=required_environment("APIFY_API_TOKEN"),
        tavily_api_key=required_environment("TAVILY_API_KEY"),
        actor_input_json=json.dumps(actor_input) if actor_input else None,
        price_actor_input_json=price_actor_input_json,
        lookback_days=lookback_days,
    )


def _tweet_price_coverage(database_path: str, ticker: str) -> dict[str, Any]:
    connection = connect_database(database_path)
    try:
        tweet_range = connection.execute(
            "SELECT MIN(market_date), MAX(market_date) FROM tweets WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        price_range = connection.execute(
            "SELECT MIN(market_date), MAX(market_date) FROM price_bars WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        overlap_days = connection.execute(
            """SELECT COUNT(DISTINCT t.market_date)
            FROM tweets t JOIN price_bars p
              ON p.ticker = t.ticker AND p.market_date = t.market_date
            WHERE t.ticker = ?""",
            (ticker,),
        ).fetchone()[0]
    finally:
        connection.close()
    return {
        "tweet_start": tweet_range[0],
        "tweet_end": tweet_range[1],
        "price_start": price_range[0],
        "price_end": price_range[1],
        "overlap_days": overlap_days,
    }


def _resolve_zero_overlap(
    ticker: str,
    database_path: str,
    market_timezone: str,
    keywords: list[str],
    history_counts: tuple[int, int, int],
    prompt_user: bool,
) -> tuple[tuple[int, int, int], dict[str, Any], str, list[str]]:
    coverage = _tweet_price_coverage(database_path, ticker)
    decision = "matched_existing_data" if coverage["overlap_days"] else "proceeded_without_overlap"
    additional_files: list[str] = []
    while coverage["overlap_days"] == 0 and prompt_user:
        tweet_range = (
            f"{coverage['tweet_start']} to {coverage['tweet_end']}"
            if coverage["tweet_start"]
            else "no tweet dates found"
        )
        price_range = (
            f"{coverage['price_start']} to {coverage['price_end']}"
            if coverage["price_start"]
            else "no price dates found"
        )
        print(
            f"\nNo ticker-matched tweet dates overlap price dates for {ticker}. "
            f"Tweet dates: {tweet_range}; price dates: {price_range}.\n"
            "With zero overlap, tweets cannot contribute sentiment to the forecast. "
            "The chart will show no observed sentiment on the available price dates, "
            "and the forecast relies on price-derived features only."
        )
        try:
            answer = input(
                "Enter a different/additional tweet CSV path, type P to proceed with current data, or Q to quit: "
            ).strip()
        except EOFError as exc:
            raise RuntimeError(
                "No interactive input is available. Provide matching data, or use "
                "--proceed-without-overlap to continue without the prompt."
            ) from exc

        if answer.casefold() in {"p", "proceed"}:
            decision = "proceeded_without_overlap"
            break
        if answer.casefold() in {"q", "quit", "exit"}:
            raise ValueError("Pipeline cancelled because tweet and price dates do not overlap")
        if not answer:
            print("Enter a CSV path, P to proceed, or Q to quit.")
            continue

        alternate_file = Path(answer).expanduser()
        if not alternate_file.is_file():
            print(f"CSV file not found: {alternate_file}")
            continue
        additional_counts = import_history(
            str(alternate_file),
            ticker,
            database_path,
            market_timezone,
            search_terms=[ticker, f"${ticker}", *keywords],
        )
        history_counts = tuple(
            previous + current for previous, current in zip(history_counts, additional_counts)
        )
        additional_files.append(str(alternate_file.resolve()))
        coverage = _tweet_price_coverage(database_path, ticker)
        if coverage["overlap_days"]:
            decision = "alternate_file_added_overlap"
        if coverage["overlap_days"] == 0:
            print("That file still has no dates matching the stored price bars. Try another file or choose P to proceed.")

    return history_counts, coverage, decision, additional_files


def _data_quality_note(
    coverage: dict[str, Any],
    overlap_decision: str,
    additional_files: list[str],
) -> str:
    if coverage["overlap_days"] == 0:
        tweet_range = f"{coverage['tweet_start']} to {coverage['tweet_end']}" if coverage["tweet_start"] else "unavailable"
        price_range = f"{coverage['price_start']} to {coverage['price_end']}" if coverage["price_start"] else "unavailable"
        choice_label = (
            "The run proceeded with insufficient date overlap"
            if overlap_decision == "proceeded_without_overlap"
            else "No overlapping data was available"
        )
        note = (
            f"{choice_label}. Tweet dates were {tweet_range}; price dates were {price_range}; "
            "there were zero aligned tweet/price dates. Consequently, tweet sentiment could not contribute "
            "to any daily feature row. Model metrics describe price-derived inputs only and are not evidence "
            "that X sentiment predicts this ticker. The chart pairs prices with neutral/no-post sentiment values, "
            "not observed sentiment."
        )
    else:
        note = (
            f"The pipeline found {coverage['overlap_days']} distinct dates with both tweets and price bars. "
            "Features, forecast comparison, sentiment summary, and chart were generated from the final imported data."
        )

    if additional_files:
        files_text = ", ".join(additional_files)
        note += (
            f" Additional CSV file(s) supplied during the overlap prompt: {files_text}. "
            "Those records were imported before downstream features and reports were regenerated."
        )
    return note


def run_pipeline(
    ticker: str,
    data_file: str | Path | None,
    database_path: str | Path,
    market_timezone: str,
    keywords: list[str],
    output_dir: str | Path,
    max_items: int | None = None,
    run_live_ingestion: bool = True,
    prompt_on_no_overlap: bool = True,
    lookback_days: int = 180,
) -> dict[str, Any]:
    load_environment()
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty")

    source = Path(data_file).expanduser() if data_file else None
    if source is not None and not source.is_file():
        raise FileNotFoundError(f"Tweet CSV file not found: {source}")
    if lookback_days not in {30, 60, 90, 180}:
        raise ValueError("lookback_days must be one of 30, 60, 90, or 180")

    database = str(database_path)
    if run_live_ingestion:
        live_counts = _run_recent_ingestion(
            ticker, keywords, database, market_timezone, max_items, lookback_days
        )
    else:
        live_counts = (0, 0, 0)

    history_counts = (0, 0, 0)
    if source is not None:
        history_counts = import_history(
            str(source),
            ticker,
            database,
            market_timezone,
            search_terms=[ticker, f"${ticker}", *keywords],
        )
    history_counts, coverage, overlap_decision, additional_files = _resolve_zero_overlap(
        ticker,
        database,
        market_timezone,
        keywords,
        history_counts,
        prompt_on_no_overlap,
    )

    connection = connect_database(database)
    try:
        features = build_daily_features(connection, ticker)
        persist_daily_features(connection, features, ticker)
    finally:
        connection.close()

    output = Path(output_dir)
    sentiment_path = output / f"{ticker.lower()}_sentiment_summary.md"
    forecast_path = output / f"{ticker.lower()}_forecast_comparison.md"
    evaluation_path = output / f"{ticker.lower()}_risk_evaluation.md"
    chart_path = output / f"{ticker.lower()}_sentiment_vs_price.png"

    data_quality_note = _data_quality_note(coverage, overlap_decision, additional_files)

    write_sentiment_summary(features, ticker, sentiment_path, data_quality_note=data_quality_note)
    generate_sentiment_price_chart(features, chart_path)
    model_summary = build_horizon_summary(features)
    write_model_summary(ticker, model_summary, forecast_path, data_quality_note=data_quality_note)
    risk_metrics = write_risk_evaluation_report(
        ticker=ticker,
        features=features,
        output_path=evaluation_path,
    )

    return {
        "ticker": ticker,
        "live_counts": live_counts,
        "history_counts": history_counts,
        "feature_rows": len(features),
        "overlap_days": coverage["overlap_days"],
        "overlap_decision": overlap_decision,
        "additional_data_files": additional_files,
        "sentiment_report": sentiment_path,
        "forecast_report": forecast_path,
        "evaluation_report": evaluation_path,
        "risk_metrics": risk_metrics,
        "chart": chart_path,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a tweet CSV, ingest current sources, build features, compare models, and chart sentiment vs. price."
    )
    parser.add_argument("--ticker", required=True, help="Asset ticker, for example AMZN or BTC-USD")
    parser.add_argument("--data-file", help="Optional historical tweet CSV to supplement recent X scraping")
    parser.add_argument("--keyword", action="append", default=[], help="Additional tweet text filter/search term; repeat as needed")
    parser.add_argument("--database", default=_default_database_path())
    parser.add_argument("--market-timezone", default=os.getenv("MARKET_TIMEZONE", "America/New_York"))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-items", type=_positive_int, help="Maximum recent X results; may increase Apify charges")
    parser.add_argument(
        "--days",
        type=int,
        choices=(30, 60, 90, 180),
        default=180,
        help="Lookback for recent X posts, prices, and news; defaults to assignment-required 180 days",
    )
    parser.add_argument(
        "--skip-live-ingestion",
        action="store_true",
        help="Use price bars already in the database; no Apify or Tavily requests are made",
    )
    parser.add_argument(
        "--proceed-without-overlap",
        action="store_true",
        help="Skip the zero-overlap prompt and run with the available price-only signal",
    )
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = _parse_args()
    try:
        result = run_pipeline(
            ticker=args.ticker,
            data_file=args.data_file,
            database_path=args.database,
            market_timezone=args.market_timezone,
            keywords=args.keyword,
            output_dir=args.output_dir,
            max_items=args.max_items,
            run_live_ingestion=not args.skip_live_ingestion,
            prompt_on_no_overlap=not args.proceed_without_overlap,
            lookback_days=args.days,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Pipeline stopped: {exc}") from None

    live = result["live_counts"]
    history = result["history_counts"]
    print(f"Live ingestion: {live[0]} X posts, {live[1]} price bars, {live[2]} news results")
    print(f"Historical CSV: {history[0]} imported/updated, {history[1]} filtered, {history[2]} invalid")
    print(f"Built {result['feature_rows']} feature rows; tweet/price overlap: {result['overlap_days']} dates")
    if result["overlap_days"] == 0:
        print("Warning: no tweet dates overlap stored prices; sentiment cannot contribute to this forecast or chart.")
    print(f"Sentiment report: {result['sentiment_report']}")
    print(f"Forecast comparison: {result['forecast_report']}")
    print(f"Risk evaluation report: {result['evaluation_report']}")
    print(f"Sentiment/price chart: {result['chart']}")


if __name__ == "__main__":
    main()