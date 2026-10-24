CREATE UNIQUE INDEX players_fpl_id ON players ((metadata->>'fpl_id')) WHERE metadata ? 'fpl_id';
CREATE INDEX idx_availability_player ON player_availability (player_id, observed_at DESC);
