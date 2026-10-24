-- Context layer (players, lineups, availability, news, audit). Schema is ready; feeds are not connected yet (see README).
CREATE TABLE players (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), sport_id UUID NOT NULL REFERENCES sports, name TEXT NOT NULL, date_of_birth DATE, nationality TEXT,
  position TEXT, current_team_id UUID REFERENCES teams, metadata JSONB NOT NULL DEFAULT '{}');
CREATE TABLE lineups (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), match_id UUID NOT NULL REFERENCES matches, team_id UUID NOT NULL REFERENCES teams,
  status TEXT NOT NULL CHECK (status IN ('projected','probable','confirmed','superseded')), formation TEXT, players JSONB NOT NULL,
  source_id UUID REFERENCES data_sources, observed_at TIMESTAMPTZ NOT NULL);
CREATE TABLE player_availability (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), player_id UUID NOT NULL REFERENCES players, team_id UUID NOT NULL REFERENCES teams,
  status TEXT NOT NULL CHECK (status IN ('available','doubtful','injured','suspended','international_duty','returning','minutes_restricted','unknown')),
  reason TEXT, expected_return DATE, confidence NUMERIC(4,3) CHECK (confidence BETWEEN 0 AND 1), source_id UUID REFERENCES data_sources, observed_at TIMESTAMPTZ NOT NULL);
CREATE TABLE news_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(), title TEXT NOT NULL, summary TEXT, source_id UUID REFERENCES data_sources, published_at TIMESTAMPTZ, observed_at TIMESTAMPTZ NOT NULL,
  event_type TEXT NOT NULL, team_id UUID REFERENCES teams, player_id UUID REFERENCES players, sentiment NUMERIC(4,3), relevance NUMERIC(4,3), confidence NUMERIC(4,3),
  verification_status TEXT NOT NULL DEFAULT 'unverified' CHECK (verification_status IN ('unverified','corroborated','official','conflicted','rejected')),
  potential_impact JSONB NOT NULL DEFAULT '{}', raw_content_hash TEXT NOT NULL UNIQUE, injection_flagged BOOLEAN NOT NULL DEFAULT FALSE);
CREATE TABLE audit_logs (id BIGSERIAL PRIMARY KEY, actor_type TEXT NOT NULL, actor_id TEXT, action TEXT NOT NULL, entity_type TEXT, entity_id UUID, metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now());
CREATE TRIGGER immut_audit BEFORE UPDATE OR DELETE ON audit_logs FOR EACH ROW EXECUTE FUNCTION forbid_mutation();
