#!/usr/bin/env python3
"""
Populate defender_game_stats from nflfastR play-by-play data

This script extracts individual defender statistics from play-by-play data:
- Interceptions
- Sacks (full and half)
- Pass defenses / pass breakups
- Tackles (solo and assisted)
- Forced fumbles and fumble recoveries
- Tackle for loss

Data source: nfl_data_py.import_pbp_data() from the nflfastR project

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import pandas as pd

# Database path
SCRIPT_DIR = Path(__file__).parent
PHASE1_DIR = SCRIPT_DIR.parent
DATABASE_PATH = PHASE1_DIR / "database" / "nfl_data.db"


def get_connection():
    """Get database connection."""
    return sqlite3.connect(str(DATABASE_PATH))


def create_defender_stats_table():
    """Create defender_game_stats table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS defender_game_stats (
            player_id TEXT,
            game_id TEXT,
            season INTEGER,
            week INTEGER,
            team_id TEXT,
            opponent_team_id TEXT,

            -- Interceptions
            interceptions INTEGER DEFAULT 0,
            interception_yards INTEGER DEFAULT 0,
            interception_tds INTEGER DEFAULT 0,

            -- Sacks
            sacks REAL DEFAULT 0,  -- Can be 0.5 for half sacks
            sack_yards REAL DEFAULT 0,

            -- Pass defense
            passes_defended INTEGER DEFAULT 0,

            -- Tackles
            tackles_solo INTEGER DEFAULT 0,
            tackles_assist INTEGER DEFAULT 0,
            tackles_combined INTEGER DEFAULT 0,
            tackles_for_loss INTEGER DEFAULT 0,

            -- Turnovers
            forced_fumbles INTEGER DEFAULT 0,
            fumble_recoveries INTEGER DEFAULT 0,
            fumble_recovery_tds INTEGER DEFAULT 0,

            -- QB hits
            qb_hits INTEGER DEFAULT 0,

            -- Metadata
            player_name TEXT,
            position TEXT,
            ingested_at TEXT,

            PRIMARY KEY (player_id, game_id)
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_defender_stats_season_week
        ON defender_game_stats(season, week)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_defender_stats_team
        ON defender_game_stats(team_id)
    """)

    conn.commit()
    conn.close()
    print("Created defender_game_stats table")


def load_pbp_data(seasons: List[int]) -> pd.DataFrame:
    """Load play-by-play data for specified seasons."""
    try:
        import nfl_data_py as nfl
    except ImportError:
        raise ImportError("nfl_data_py is required. Install with: pip install nfl_data_py")

    print(f"Loading play-by-play data for seasons: {seasons}")
    pbp = nfl.import_pbp_data(seasons)
    print(f"Loaded {len(pbp):,} plays")
    return pbp


def aggregate_defender_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate defender statistics from play-by-play data.

    Each defender action is tracked in specific PBP columns with player IDs.
    We aggregate these per player per game.
    """
    print("Aggregating defender stats from play-by-play...")

    # Initialize stats dictionary: (player_id, game_id) -> stats dict
    stats: Dict[Tuple[str, str], Dict] = defaultdict(lambda: {
        'interceptions': 0,
        'interception_yards': 0,
        'interception_tds': 0,
        'sacks': 0.0,
        'sack_yards': 0.0,
        'passes_defended': 0,
        'tackles_solo': 0,
        'tackles_assist': 0,
        'tackles_for_loss': 0,
        'forced_fumbles': 0,
        'fumble_recoveries': 0,
        'fumble_recovery_tds': 0,
        'qb_hits': 0,
        'season': None,
        'week': None,
        'team_id': None,
        'player_name': None,
        'position': None,
    })

    # Track player info for each player_id
    player_info: Dict[str, Dict] = {}

    # Filter to actual plays (not timeouts, end of quarter, etc.)
    plays = pbp[pbp['play_type'].notna()].copy()

    # Process interceptions
    int_plays = plays[plays['interception'] == 1]
    for _, row in int_plays.iterrows():
        player_id = row.get('interception_player_id')
        if pd.notna(player_id):
            game_id = row['game_id']
            key = (player_id, game_id)
            stats[key]['interceptions'] += 1
            stats[key]['interception_yards'] += row.get('return_yards', 0) or 0
            if row.get('return_touchdown') == 1:
                stats[key]['interception_tds'] += 1
            stats[key]['season'] = row['season']
            stats[key]['week'] = row['week']
            stats[key]['team_id'] = row.get('defteam')
            player_info[player_id] = {
                'name': row.get('interception_player_name'),
            }

    print(f"  Processed {len(int_plays):,} interception plays")

    # Process sacks
    sack_plays = plays[plays['sack'] == 1]
    for _, row in sack_plays.iterrows():
        game_id = row['game_id']
        yards_lost = abs(row.get('yards_gained', 0) or 0)

        # Full sack
        sack_player = row.get('sack_player_id')
        if pd.notna(sack_player):
            key = (sack_player, game_id)
            stats[key]['sacks'] += 1.0
            stats[key]['sack_yards'] += yards_lost
            stats[key]['season'] = row['season']
            stats[key]['week'] = row['week']
            stats[key]['team_id'] = row.get('defteam')
            player_info[sack_player] = {'name': row.get('sack_player_name')}

        # Half sacks
        for half_sack_col in ['half_sack_1_player_id', 'half_sack_2_player_id']:
            half_sack_player = row.get(half_sack_col)
            if pd.notna(half_sack_player):
                key = (half_sack_player, game_id)
                stats[key]['sacks'] += 0.5
                stats[key]['sack_yards'] += yards_lost / 2
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')

    print(f"  Processed {len(sack_plays):,} sack plays")

    # Process pass defenses
    for _, row in plays.iterrows():
        game_id = row['game_id']
        for pd_col in ['pass_defense_1_player_id', 'pass_defense_2_player_id']:
            pd_player = row.get(pd_col)
            if pd.notna(pd_player):
                key = (pd_player, game_id)
                stats[key]['passes_defended'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')
                name_col = pd_col.replace('_id', '_name')
                player_info[pd_player] = {'name': row.get(name_col)}

    # Process solo tackles
    for _, row in plays.iterrows():
        game_id = row['game_id']
        for tackle_col in ['solo_tackle_1_player_id', 'solo_tackle_2_player_id']:
            tackle_player = row.get(tackle_col)
            if pd.notna(tackle_player):
                key = (tackle_player, game_id)
                stats[key]['tackles_solo'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')
                name_col = tackle_col.replace('_id', '_name')
                player_info[tackle_player] = {'name': row.get(name_col)}

    # Process assist tackles
    for _, row in plays.iterrows():
        game_id = row['game_id']
        for assist_col in ['assist_tackle_1_player_id', 'assist_tackle_2_player_id',
                           'assist_tackle_3_player_id', 'assist_tackle_4_player_id']:
            assist_player = row.get(assist_col)
            if pd.notna(assist_player):
                key = (assist_player, game_id)
                stats[key]['tackles_assist'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')

    # Process tackle for loss
    tfl_plays = plays[plays['tackled_for_loss'] == 1]
    for _, row in tfl_plays.iterrows():
        game_id = row['game_id']
        for tfl_col in ['tackle_for_loss_1_player_id', 'tackle_for_loss_2_player_id']:
            tfl_player = row.get(tfl_col)
            if pd.notna(tfl_player):
                key = (tfl_player, game_id)
                stats[key]['tackles_for_loss'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')

    print(f"  Processed tackle plays")

    # Process forced fumbles
    for _, row in plays.iterrows():
        game_id = row['game_id']
        for ff_col in ['forced_fumble_player_1_id', 'forced_fumble_player_2_id']:
            ff_player = row.get(ff_col)
            if pd.notna(ff_player):
                key = (ff_player, game_id)
                stats[key]['forced_fumbles'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')

    # Process fumble recoveries
    fumble_plays = plays[plays['fumble_lost'] == 1]
    for _, row in fumble_plays.iterrows():
        game_id = row['game_id']
        for fr_col in ['fumble_recovery_1_player_id', 'fumble_recovery_2_player_id']:
            fr_player = row.get(fr_col)
            if pd.notna(fr_player):
                # Only count if recovered by defense
                recovery_team = row.get('fumble_recovery_1_team') if fr_col == 'fumble_recovery_1_player_id' else row.get('fumble_recovery_2_team')
                posteam = row.get('posteam')
                if recovery_team != posteam:  # Defense recovered
                    key = (fr_player, game_id)
                    stats[key]['fumble_recoveries'] += 1
                    stats[key]['season'] = row['season']
                    stats[key]['week'] = row['week']
                    stats[key]['team_id'] = row.get('defteam')
                    # Check for TD
                    td_col = 'fumble_recovery_1_yards' if fr_col == 'fumble_recovery_1_player_id' else 'fumble_recovery_2_yards'
                    if row.get('return_touchdown') == 1:
                        stats[key]['fumble_recovery_tds'] += 1

    print(f"  Processed fumble plays")

    # Process QB hits
    for _, row in plays.iterrows():
        game_id = row['game_id']
        for qbh_col in ['qb_hit_1_player_id', 'qb_hit_2_player_id']:
            qbh_player = row.get(qbh_col)
            if pd.notna(qbh_player):
                key = (qbh_player, game_id)
                stats[key]['qb_hits'] += 1
                stats[key]['season'] = row['season']
                stats[key]['week'] = row['week']
                stats[key]['team_id'] = row.get('defteam')

    print(f"  Processed QB hit plays")

    # Convert to DataFrame
    records = []
    for (player_id, game_id), stat_dict in stats.items():
        if stat_dict['season'] is None:
            continue  # Skip if no game metadata

        record = {
            'player_id': player_id,
            'game_id': game_id,
            'season': int(stat_dict['season']),
            'week': int(stat_dict['week']),
            'team_id': stat_dict['team_id'],
            'interceptions': stat_dict['interceptions'],
            'interception_yards': stat_dict['interception_yards'],
            'interception_tds': stat_dict['interception_tds'],
            'sacks': stat_dict['sacks'],
            'sack_yards': stat_dict['sack_yards'],
            'passes_defended': stat_dict['passes_defended'],
            'tackles_solo': stat_dict['tackles_solo'],
            'tackles_assist': stat_dict['tackles_assist'],
            'tackles_combined': stat_dict['tackles_solo'] + stat_dict['tackles_assist'],
            'tackles_for_loss': stat_dict['tackles_for_loss'],
            'forced_fumbles': stat_dict['forced_fumbles'],
            'fumble_recoveries': stat_dict['fumble_recoveries'],
            'fumble_recovery_tds': stat_dict['fumble_recovery_tds'],
            'qb_hits': stat_dict['qb_hits'],
            'player_name': player_info.get(player_id, {}).get('name'),
        }
        records.append(record)

    df = pd.DataFrame(records)
    print(f"Aggregated stats for {len(df):,} player-game combinations")
    return df


def populate_defender_stats(seasons: List[int] = None):
    """Main function to populate defender_game_stats table."""
    if seasons is None:
        # Default to recent seasons
        seasons = [2021, 2022, 2023, 2024]

    # Create table
    create_defender_stats_table()

    # Load PBP data
    pbp = load_pbp_data(seasons)

    # Aggregate stats
    df = aggregate_defender_stats(pbp)

    if df.empty:
        print("No defender stats to insert")
        return 0

    # Add metadata
    df['ingested_at'] = datetime.utcnow().isoformat()

    # Insert into database
    conn = get_connection()

    # Clear existing data for these seasons
    for season in seasons:
        conn.execute("DELETE FROM defender_game_stats WHERE season = ?", (season,))

    # Insert new data
    df.to_sql('defender_game_stats', conn, if_exists='append', index=False)
    conn.commit()

    # Verify
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM defender_game_stats")
    total = cursor.fetchone()[0]

    print(f"\nInserted {len(df):,} defender-game records")
    print(f"Total defender_game_stats rows: {total:,}")

    # Sample stats
    print("\nSample data:")
    cursor.execute("""
        SELECT player_name, season, week, team_id, sacks, interceptions, tackles_combined
        FROM defender_game_stats
        WHERE sacks > 0 OR interceptions > 0
        ORDER BY season DESC, week DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: S{row[1]} W{row[2]} {row[3]} - {row[4]} sacks, {row[5]} INT, {row[6]} tackles")

    # Top performers
    print("\nTop sack performers:")
    cursor.execute("""
        SELECT player_name, SUM(sacks) as total_sacks, COUNT(*) as games
        FROM defender_game_stats
        GROUP BY player_id
        ORDER BY total_sacks DESC
        LIMIT 5
    """)
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} sacks in {row[2]} games")

    conn.close()
    return len(df)


if __name__ == "__main__":
    print("=" * 60)
    print("Populate Defender Game Stats from Play-by-Play")
    print("=" * 60)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seasons', type=int, nargs='+', default=[2021, 2022, 2023, 2024],
                        help='Seasons to process')
    args = parser.parse_args()

    count = populate_defender_stats(seasons=args.seasons)
    print(f"\nDone! Inserted {count:,} records")
