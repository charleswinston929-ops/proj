-- Core schema. Typed relational tables; JSONB only for genuinely variable payloads.
CREATE TABLE sports (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL UNIQUE, code TEXT NOT NULL UNIQUE);
CREATE TABLE data_sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL UNIQUE, provider_type TEXT NOT NULL,
  tier SMALLINT NOT NULL CHECK (tier BETWEEN 1 AND 5), reliability_score NUMERIC(4,3) NOT NULL CHECK (reliability_score BETWEEN 0 AND 1),
  api_base_url TEXT, active BOOLEAN NOT NULL DEFAULT TRUE);
CREATE TABLE competitions (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), sport_id UUID NOT NULL REFERENCES sports, name TEXT NOT NULL, country TEXT, code TEXT, UNIQUE (sport_id, code));
CREATE TABLE seasons (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), competition_id UUID NOT NULL REFERENCES competitions, name TEXT NOT NULL, start_date DATE, end_date DATE, UNIQUE (competition_id, name));
CREATE TABLE teams (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), sport_id UUID NOT NULL REFERENCES sports, canonical_key TEXT NOT NULL, name TEXT NOT NULL,
  country TEXT, metadata JSONB NOT NULL DEFAULT '{}', UNIQUE (sport_id, canonical_key));
CREATE TABLE entity_aliases (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), entity_type TEXT NOT NULL CHECK (entity_type IN ('team','player','competition')),
  entity_id UUID NOT NULL, source_id UUID REFERENCES data_sources, alias TEXT NOT NULL, UNIQUE (entity_type, source_id, alias));
CREATE TABLE matches (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), competition_id UUID NOT NULL REFERENCES competitions, season_id UUID NOT NULL REFERENCES seasons,
  home_team_id UUID NOT NULL REFERENCES teams, away_team_id UUID NOT NULL REFERENCES teams, matchday INT,
  kickoff_at TIMESTAMPTZ NOT NULL, status TEXT NOT NULL CHECK (status IN ('scheduled','awaiting_result','finished','postponed','cancelled')),
  home_score INT CHECK (home_score >= 0), away_score INT CHECK (away_score >= 0), ht_home INT, ht_away INT,
  source_id UUID REFERENCES data_sources, natural_key TEXT NOT NULL UNIQUE, metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (home_team_id <> away_team_id), CHECK ((status = 'finished') = (home_score IS NOT NULL AND away_score IS NOT NULL)),
  UNIQUE (season_id, home_team_id, away_team_id));
CREATE INDEX idx_matches_kickoff ON matches (kickoff_at); CREATE INDEX idx_matches_status ON matches (status, kickoff_at);
CREATE INDEX idx_matches_home ON matches (home_team_id); CREATE INDEX idx_matches_away ON matches (away_team_id);

CREATE TABLE model_versions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), name TEXT NOT NULL, version TEXT NOT NULL, model_type TEXT NOT NULL,
  training_start DATE, training_end DATE, feature_version TEXT NOT NULL, hyperparameters JSONB NOT NULL DEFAULT '{}', metrics JSONB NOT NULL DEFAULT '{}',
  code_commit TEXT, status TEXT NOT NULL DEFAULT 'candidate' CHECK (status IN ('candidate','validated','active','retired')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE (name, version));
CREATE UNIQUE INDEX one_active_model ON model_versions ((TRUE)) WHERE status = 'active';

CREATE TABLE feature_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), match_id UUID NOT NULL REFERENCES matches, snapshot_at TIMESTAMPTZ NOT NULL, data_cutoff TIMESTAMPTZ NOT NULL,
  feature_version TEXT NOT NULL, features JSONB NOT NULL, data_quality_score NUMERIC(5,4) CHECK (data_quality_score BETWEEN 0 AND 1),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(), CHECK (data_cutoff <= snapshot_at));
CREATE TABLE prediction_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), match_id UUID NOT NULL REFERENCES matches, kind TEXT NOT NULL CHECK (kind IN ('live','reconstructed')),
  prediction_at TIMESTAMPTZ NOT NULL, prediction_horizon TEXT NOT NULL, model_version_id UUID NOT NULL REFERENCES model_versions,
  feature_snapshot_id UUID REFERENCES feature_snapshots, data_cutoff TIMESTAMPTZ NOT NULL,
  home_probability NUMERIC(7,6) NOT NULL CHECK (home_probability BETWEEN 0 AND 1), draw_probability NUMERIC(7,6) NOT NULL CHECK (draw_probability BETWEEN 0 AND 1),
  away_probability NUMERIC(7,6) NOT NULL CHECK (away_probability BETWEEN 0 AND 1),
  expected_home_goals NUMERIC(6,4), expected_away_goals NUMERIC(6,4), confidence_level TEXT NOT NULL, confidence_score NUMERIC(5,4),
  model_disagreement NUMERIC(6,5), data_quality_score NUMERIC(5,4) NOT NULL, simulation_count INT, components JSONB NOT NULL, intervals JSONB, scorelines JSONB, explanation JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (abs(home_probability + draw_probability + away_probability - 1) < 0.00001), CHECK (data_cutoff <= prediction_at),
  UNIQUE (match_id, model_version_id, prediction_at, kind));
CREATE INDEX idx_pred_match ON prediction_snapshots (match_id, prediction_at);
CREATE TABLE prediction_lineage (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), prediction_id UUID NOT NULL REFERENCES prediction_snapshots, previous_prediction_id UUID REFERENCES prediction_snapshots,
  event_type TEXT NOT NULL, source_entity_type TEXT, source_entity_id UUID, previous_value JSONB, new_value JSONB, impact_score NUMERIC(7,5), explanation TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TABLE team_strength_snapshots (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), team_id UUID NOT NULL REFERENCES teams, as_of TIMESTAMPTZ NOT NULL, model_version_id UUID NOT NULL REFERENCES model_versions,
  elo NUMERIC(8,3), attack NUMERIC(8,5), defence NUMERIC(8,5), UNIQUE (team_id, as_of, model_version_id));
CREATE TABLE backtest_predictions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), match_id UUID NOT NULL REFERENCES matches, model_version_id UUID NOT NULL REFERENCES model_versions,
  p_home NUMERIC(7,6) NOT NULL, p_draw NUMERIC(7,6) NOT NULL, p_away NUMERIC(7,6) NOT NULL, components JSONB NOT NULL, disagreement NUMERIC(6,5), UNIQUE (match_id, model_version_id));
CREATE TABLE model_evaluations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), model_version_id UUID NOT NULL REFERENCES model_versions, window_name TEXT NOT NULL,
  period_start DATE NOT NULL, period_end DATE NOT NULL, sample_size INT NOT NULL, log_loss NUMERIC(8,5), brier_score NUMERIC(8,5), rps NUMERIC(8,5),
  accuracy NUMERIC(6,5), calibration_error NUMERIC(8,5), metrics JSONB NOT NULL DEFAULT '{}', created_at TIMESTAMPTZ NOT NULL DEFAULT now());

-- Immutability (spec 21/64): history can never be rewritten, even by a buggy job.
CREATE FUNCTION forbid_mutation() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION '% is append-only (% blocked)', TG_TABLE_NAME, TG_OP; END $$;
CREATE TRIGGER immut_pred BEFORE UPDATE OR DELETE ON prediction_snapshots FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
CREATE TRIGGER immut_feat BEFORE UPDATE OR DELETE ON feature_snapshots FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
CREATE TRIGGER immut_lineage BEFORE UPDATE OR DELETE ON prediction_lineage FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
CREATE TRIGGER immut_backtest BEFORE UPDATE OR DELETE ON backtest_predictions FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
