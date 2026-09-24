# Verified regional event cards

The dashboard reads curated post-event records from
`data/dashboard/verified_events.csv`. These cards are retrospective public
information, not operational reporting or forecasting.

## Required fields

- `event_id`: stable unique identifier.
- `date`: publication/event date in ISO format.
- `oblast`: normalized oblast name used by the dashboard.
- `title`, `attack_types`, `summary`, `damage`, `casualties`: concise Ukrainian
  summaries that do not add claims beyond the source.
- `source_name`, `source_url`: official OVA, local authority, or DSNS page.
- `image_urls`, `image_alts`: pipe-separated official images and matching alt
  text. Leave both empty when the official publication has no suitable image.
- `image_license`: the source's stated license or usage note.
- `is_verified`: `true` only after a human checked the page, date, oblast,
  summary, and images.

## Publication rules

1. Use only post-event official publications. Do not include live routes,
   launch coordinates, exact operational timing, or unpublished locations.
2. Do not show bodies, severe injuries, identifiable victims, or children in
   distress. Prefer wide aftermath, fire-response, and building-damage images.
3. Link every card to the source. Never infer a weapon type, casualty count,
   or damage category that the source does not state.
4. A missing card means no verified publication was added to the snapshot; it
   never means that no attack occurred.
5. Keep at most three suitable images per event and preserve attribution.

Run `pytest tests/test_verified_events.py` after editing the feed.
