from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any

import requests
from apify_client import ApifyClient


def _replace_template_values(value: Any, values: dict[str, Any]) -> Any:
    if isinstance(value, str):
        if value == "$SEARCH_TERMS":
            return values["SEARCH_TERMS"]
        for key, replacement in values.items():
            if key == "SEARCH_TERMS":
                continue
            value = value.replace(f"${key}", replacement)
        return value
    if isinstance(value, list):
        return [_replace_template_values(item, values) for item in value]
    if isinstance(value, dict):
        return {key: _replace_template_values(item, values) for key, item in value.items()}
    return value


def _default_dataset_id(run: Any) -> str | None:
    if isinstance(run, dict):
        return run.get("defaultDatasetId") or run.get("default_dataset_id")
    return getattr(run, "default_dataset_id", None) or getattr(run, "defaultDatasetId", None)


def fetch_x_posts(
    ticker: str,
    keywords: list[str],
    start_date: date,
    end_date: date,
    actor_id: str,
    api_token: str,
    actor_input_json: str | None = None,
) -> list[dict[str, Any]]:
    terms = [ticker.upper(), f"${ticker.upper()}", *keywords]
    unique_terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))
    date_filter = f"since:{start_date.isoformat()} until:{end_date.isoformat()}"
    search_terms = [f'"{term}" {date_filter}' for term in unique_terms]
    values = {
        "TICKER": ticker.upper(),
        "CASHTAG": f"${ticker.upper()}",
        "KEYWORDS": " ".join(keywords),
        "SEARCH_TERMS": search_terms,
        "START_DATE": start_date.isoformat(),
        "END_DATE": end_date.isoformat(),
        "DATE_FILTER": date_filter,
    }

    if actor_input_json:
        template = json.loads(actor_input_json)
        actor_input = _replace_template_values(template, values)
        actor_input["searchTerms"] = search_terms
    else:
        actor_input = {"searchTerms": search_terms}

    client = ApifyClient(api_token)
    run = client.actor(actor_id).call(run_input=actor_input)
    dataset_id = _default_dataset_id(run)
    if not dataset_id:
        raise RuntimeError("Apify actor completed without a default dataset ID")
    records = list(client.dataset(dataset_id).iterate_items())
    if not records:
        raise RuntimeError(
            "Apify X Actor returned no posts. Check the run log for account or monthly limits, "
            "and verify the Actor's access plan before retrying."
        )
    return records


def fetch_price_history(
    ticker: str,
    start_date: date,
    end_date: date,
    actor_id: str,
    api_token: str,
    actor_input_json: str,
) -> list[dict[str, Any]]:
    if not actor_input_json:
        raise RuntimeError("APIFY_PRICE_ACTOR_INPUT_JSON must match the selected price Actor's input schema")
    values = {
        "TICKER": ticker.upper(),
        "START_DATE": start_date.isoformat(),
        "END_DATE": end_date.isoformat(),
    }
    actor_input = _replace_template_values(json.loads(actor_input_json), values)
    client = ApifyClient(api_token)
    run = client.actor(actor_id).call(run_input=actor_input)
    dataset_id = _default_dataset_id(run)
    if not dataset_id:
        raise RuntimeError("Apify price Actor completed without a default dataset ID")
    records = list(client.dataset(dataset_id).iterate_items())
    if not records:
        raise RuntimeError(f"Apify price Actor returned no price records for {ticker}")
    return records


def fetch_news(ticker: str, api_key: str, start_date: date, max_results: int = 20) -> list[dict[str, Any]]:
    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": f"{ticker.upper()} asset market news",
            "topic": "news",
            "search_depth": "basic",
            "start_date": start_date.isoformat(),
            "include_published_date": True,
            "max_results": max_results,
        },
        timeout=45,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def load_environment() -> None:
    from dotenv import load_dotenv

    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env", override=False)


def required_environment(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value