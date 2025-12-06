import sqlite3

conn = sqlite3.connect('../config/database/nfl_data.db')

# Drop the old table first
conn.execute("DROP TABLE IF EXISTS team_defense_game_stats")

# Create with correct schema
conn.execute("""
    CREATE TABLE IF NOT EXISTS team_defense_game_stats (
        team_id TEXT NOT NULL,
        season INTEGER NOT NULL,
        week INTEGER NOT NULL,
        game_id TEXT,
        opponent_team_id TEXT,
        total_yards_allowed REAL,
        pass_yards_allowed REAL,
        rush_yards_allowed REAL,
        total_tds_allowed REAL,
        pass_tds_allowed REAL,
        rush_tds_allowed REAL,
        wr_targets_allowed INTEGER,
        wr_yards_allowed REAL,
        wr_tds_allowed REAL,
        te_targets_allowed INTEGER,
        te_yards_allowed REAL,
        te_tds_allowed REAL,
        rb_targets_allowed INTEGER,
        rb_yards_allowed REAL,
        rb_rec_tds_allowed REAL,
        rb_rush_yards_allowed REAL,
        rb_rush_tds_allowed REAL,
        sacks REAL,
        interceptions INTEGER,
        fumbles_recovered INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (team_id, season, week)
    )
""")

conn.commit()
print("Table dropped and recreated successfully!")
conn.close()