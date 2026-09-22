# Data model

The project uses a normalized historical event model (нормалізована структура: одна подія зберігається окремо від джерел, зброї, втрат і повітряних тривог).

## Main tables

### sources
Stores provenance (походження даних):
- source name and URL
- source type
- publication date
- reliability level

### geography
Canonical geographic entities (єдині назви областей/районів/населених пунктів) and optional map coordinates.

### attacks
One normalized historical event:
- start/end time
- oblast, raion, settlement
- optional generalized geographic coordinates
- attack type
- evidence confidence
- description

### attack_sources
Many-to-many provenance link (зв'язок багато-до-багатьох):
- one event can be supported by several sources
- preserves source event ID when available
- optional source URL/hash for audit and deduplication

### weapons
Weapon category/type and reported quantities linked to an attack.

### casualties
Reported killed/injured counts and child casualty fields where available.

### alerts
Historical air-alert intervals. An alert is not treated as proof that an attack occurred.

## Data principles

1. Preserve original source and publication date.
2. Never silently convert uncertain reports into confirmed events.
3. Possible duplicates are flagged first; automatic deletion is avoided.
4. Raw data is immutable; transformations go into processed data.
5. Store timestamps in UTC internally.
6. Prefer stable geographic IDs over free-text matching when a geographic source provides them.
7. Keep multiple supporting sources instead of overwriting provenance.
8. Forecasting uses aggregated historical risk by broad geography/time windows, not exact targets, routes, launch points, or live coordinates.
