-- PostgreSQL schema for the historical attack analytics project.
-- All timestamps are stored in UTC.

CREATE TABLE IF NOT EXISTS sources (
    source_id BIGSERIAL PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_type TEXT NOT NULL,
    publication_date DATE,
    reliability_level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS attacks (
    attack_id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    oblast TEXT NOT NULL,
    raion TEXT,
    settlement TEXT,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    attack_type TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT 'reported',
    source_id BIGINT REFERENCES sources(source_id),
    description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS weapons (
    weapon_id BIGSERIAL PRIMARY KEY,
    attack_id BIGINT NOT NULL REFERENCES attacks(attack_id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    type TEXT,
    quantity INTEGER,
    intercepted_quantity INTEGER,
    source_id BIGINT REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS casualties (
    casualty_id BIGSERIAL PRIMARY KEY,
    attack_id BIGINT NOT NULL REFERENCES attacks(attack_id) ON DELETE CASCADE,
    killed INTEGER NOT NULL DEFAULT 0,
    injured INTEGER NOT NULL DEFAULT 0,
    children_killed INTEGER,
    children_injured INTEGER,
    source_id BIGINT REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    oblast TEXT,
    raion TEXT,
    settlement TEXT,
    threat_type TEXT,
    source_id BIGINT REFERENCES sources(source_id)
);

CREATE INDEX IF NOT EXISTS idx_attacks_started_at ON attacks(started_at);
CREATE INDEX IF NOT EXISTS idx_attacks_oblast ON attacks(oblast);
CREATE INDEX IF NOT EXISTS idx_attacks_type ON attacks(attack_type);
CREATE INDEX IF NOT EXISTS idx_weapons_attack_id ON weapons(attack_id);
CREATE INDEX IF NOT EXISTS idx_alerts_started_at ON alerts(started_at);
CREATE INDEX IF NOT EXISTS idx_alerts_oblast ON alerts(oblast);
