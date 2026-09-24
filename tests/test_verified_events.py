"""Validate the curated official-event feed shown in the dashboard."""

from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "data/dashboard/verified_events.csv"

REQUIRED_COLUMNS = {
    "event_id",
    "date",
    "oblast",
    "title",
    "attack_types",
    "summary",
    "damage",
    "casualties",
    "source_name",
    "source_url",
    "image_urls",
    "image_alts",
    "image_license",
    "is_verified",
}

OFFICIAL_DOMAINS = {
    "oda.od.gov.ua",
    "dn.gov.ua",
    "kharkivoda.gov.ua",
    "www.zoda.gov.ua",
}


def test_verified_events_have_official_sources_and_safe_images():
    events = pd.read_csv(EVENTS_PATH)
    assert REQUIRED_COLUMNS.issubset(events.columns)
    assert events["event_id"].is_unique
    assert events["is_verified"].astype(str).str.lower().eq("true").all()

    for _, event in events.iterrows():
        source = urlparse(event["source_url"])
        assert source.scheme == "https"
        assert source.netloc in OFFICIAL_DOMAINS
        assert event["title"] and event["summary"] and event["damage"]

        image_urls = [
            url for url in str(event.get("image_urls") or "").split("|")
            if url and url.lower() != "nan"
        ]
        image_alts = [
            alt for alt in str(event.get("image_alts") or "").split("|")
            if alt and alt.lower() != "nan"
        ]
        assert len(image_urls) == len(image_alts)
        for image_url in image_urls:
            parsed = urlparse(image_url)
            assert parsed.scheme == "https"
            assert parsed.netloc in OFFICIAL_DOMAINS
