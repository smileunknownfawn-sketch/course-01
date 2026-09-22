# Data model

The project uses a normalized historical event model. The data is intended for retrospective analysis and aggregated civilian-safety forecasting, not operational targeting.

## Core entities

### attacks
- attack_id — unique event identifier
- started_at — event start datetime (UTC)
- ended_at — event end datetime (UTC), nullable
- oblast — oblast name
- raion — district, nullable
- settlement — city/town/community, nullable
- latitude / longitude — approximate location, nullable
- attack_type — missile, UAV, KAB, artillery, other
- confidence — confirmed, probable, reported, unverified
- source_id — primary source reference
- description — short factual note

### weapons
- weapon_id — unique identifier
- attack_id — related attack
- category — missile, UAV, KAB, other
- type — specific type when documented
- quantity — nullable
- intercepted_quantity — nullable
- source_id — source reference

### casualties
- casualty_id — unique identifier
- attack_id — related attack
- killed — civilians killed
- injured — civilians injured
- children_killed — nullable
- children_injured — nullable
- source_id — source reference

### alerts
- alert_id — unique identifier
- started_at / ended_at — alert interval
- oblast / raion / settlement — affected area
- threat_type — documented alert type
- source_id — source reference

### sources
- source_id — unique identifier
- source_name
- source_url
- source_type
- publication_date
- reliability_level

## Data principles

1. Preserve the original source and publication date.
2. Never silently convert uncertain reports into confirmed events.
3. Deduplicate events using time, geography, description and source overlap.
4. Keep raw data immutable; transformations go into processed tables.
5. Store UTC internally and convert to local time only for presentation.
6. Forecasting uses aggregated historical risk windows rather than exact future targets, routes or coordinates.
