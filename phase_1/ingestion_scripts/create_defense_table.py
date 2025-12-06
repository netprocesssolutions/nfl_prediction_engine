import sqlite3

conn = sqlite3.connect('../config/database/nfl_data.db')

conn.execute("""
    CREATE TABLE IF NOT EXISTS team_defense_game_stats (
        team_id TEXT NOT NULL,
        game_id TEXT NOT NULL,
        season INTEGER NOT NULL,
        week INTEGER NOT NULL,
        opponent_team_id TEXT,
        points_allowed INTEGER,
        yards_allowed_passing REAL,
        yards_allowed_rushing REAL,
        yards_allowed_total REAL,
        yards_allowed_to_wr REAL,
        yards_allowed_to_te REAL,
        yards_allowed_to_rb REAL,
        targets_allowed_to_wr INTEGER,
        targets_allowed_to_te INTEGER,
        targets_allowed_to_rb INTEGER,
        tds_allowed_to_wr REAL,
        tds_allowed_to_te REAL,
        tds_allowed_to_rb REAL,
        redzone_defense_efficiency REAL,
        epa_allowed REAL,
        success_rate_allowed REAL,
        explosive_plays_allowed INTEGER,
        sacks REAL,
        interceptions INTEGER,
        fumbles_recovered INTEGER,
        defensive_tds INTEGER,
        raw_json TEXT,
        ingested_at TEXT,
        PRIMARY KEY (team_id, game_id)
    )
""")

conn.commit()
print("Table created successfully!")
conn.close()