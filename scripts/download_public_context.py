"""Download additional public historical context sources."""

from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
VIINA_DIR = RAW / "viina"
SIRENS_PATH = RAW / "volunteer_air_raids_uk.csv"
GEO_PATH = RAW / "ukraine_admin1.geojson"

VIINA_TREE_URL = (
    "https://api.github.com/repos/scott4ai/ukraine-violence-data/"
    "git/trees/master?recursive=1"
)
VIINA_RAW_PREFIX = (
    "https://raw.githubusercontent.com/scott4ai/ukraine-violence-data/"
    "master/"
)
SIRENS_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/main/datasets/volunteer_data_uk.csv"
)
GEO_URL = (
    "https://raw.githubusercontent.com/darmat1/ukraine-geo-data/"
    "main/geodata/Ukraine.geojson"
)

HEADERS = {"User-Agent": "course-01-historical-analytics/1.0"}


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers=HEADERS, timeout=120, stream=True) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    VIINA_DIR.mkdir(parents=True, exist_ok=True)

    tree_response = requests.get(
        VIINA_TREE_URL,
        headers=HEADERS,
        timeout=30,
    )
    tree_response.raise_for_status()
    tree = tree_response.json().get("tree", [])
    viina_paths = sorted(
        item["path"]
        for item in tree
        if item.get("type") == "blob"
        and item.get("path", "").startswith("output/monthly/viina_incidents_")
        and item.get("path", "").endswith(".csv")
    )

    for source_path in viina_paths:
        destination = VIINA_DIR / Path(source_path).name
        download(VIINA_RAW_PREFIX + source_path, destination)

    download(SIRENS_URL, SIRENS_PATH)
    download(GEO_URL, GEO_PATH)

    print(
        json.dumps(
            {
                "viina_files": len(viina_paths),
                "sirens_file": str(SIRENS_PATH),
                "geo_file": str(GEO_PATH),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
