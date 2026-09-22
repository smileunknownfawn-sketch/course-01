# Data quality (якість даних)

Before data reaches the dashboard or ML model, the pipeline checks:

1. Required columns exist.
2. Event IDs are unique within the normalized dataset.
3. Dates are valid and stored in UTC.
4. Oblast names are normalized.
5. Confidence values use the controlled vocabulary.
6. Possible duplicates are flagged for review rather than silently deleted.
7. Every factual event keeps a source reference.
8. Air-raid alerts are not treated as confirmed attacks.
9. Missing values remain missing; the pipeline does not invent facts.

## Confidence levels

- confirmed: supported by strong source evidence.
- probable: supported by multiple or sufficiently detailed reports.
- reported: publicly reported but not independently verified.
- unverified: insufficient evidence for stronger classification.

These labels describe evidence quality, not certainty about future events.
