#!/usr/bin/env python3
"""
Backfill player_game_stats from Sleeper API for historical seasons.

This script fetches offensive stats from Sleeper API for seasons
that aren't already in the database (e.g., 2021-2022).
"""

import requests
import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Database path
SCRIPT_DIR = Path(__file__).parent
DATABASE_PATH = SCRIPT_DIR.parent / "database" / "nfl_data.db"

# Sleeper API
SLEEPER_STATS_URL = "https://api.sleeper.app/v1/stats/nfl/regular/{season}/{week}"

# Offensive positions we care about
OFFENSIVE_POSITIONS = {'QB', 'RB', 'WR', 'TE', 'FB'}

# Team abbreviation mapping (Sleeper uses some different abbreviations)
TEAM_MAP = {
    'JAC': 'JAX', 'JAX': 'JAX',
    'WSH': 'WAS', 'WAS': 'WAS',
    'LA': 'LAR', 'LAR': 'LAR',
    'LV': 'LV', 'OAK': 'LV',
}


def get_connection():
    return sqlite3.connect(str(DATABASE_PATH))


def load_players(conn) -> Dict[str, dict]:
    """Load player info keyed by Sleeper ID."""
    cursor = conn.cursor()
    cursor.execute('SELECT player_id, full_name, position, team_id FROM players')
    return {str(row[0]): {'name': row[1], 'position': row[2], 'team': row[3]}
            for row in cursor.fetchall()}


def load_games(conn) -> Dict[Tuple, Tuple]:
    """Load games keyed by (season, week, team) -> (game_id, opponent)."""
    cursor = conn.cursor()
    cursor.execute('SELECT game_id, season, week, home_team_id, away_team_id FROM games')
    games = {}
    for row in cursor.fetchall():
        game_id, season, week, home, away = row
        games[(season, week, home)] = (game_id, away)
        games[(season, week, away)] = (game_id, home)
    return games


def normalize_team(team: str) -> str:
    """Normalize team abbreviation."""
    if not team:
        return None
    team = team.upper()
    return TEAM_MAP.get(team, team)


def fetch_week_stats(season: int, week: int) -> Optional[dict]:
    """Fetch stats from Sleeper API."""
    url = SLEEPER_STATS_URL.format(season=season, week=week)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 404:
            return None
        else:
            print(f"  API error: {resp.status_code}")
            return None
    except requests.RequestException as e:
        print(f"  Request failed: {e}")
        return None


def parse_player_stats(player_id: str, stats: dict, player_info: dict,
                       season: int, week: int, games: dict) -> Optional[dict]:
    """Parse Sleeper stats into player_game_stats format."""

    # Get player's team from stats or player info
    team = normalize_team(stats.get('team') or player_info.get('team'))
    if not team:
        return None

    position = player_info.get('position', '')
    if position not in OFFENSIVE_POSITIONS:
        return None

    # Find game_id and opponent
    game_key = (season, week, team)
    if game_key not in games:
        return None

    game_id, opponent = games[game_key]

    # Extract stats
    return {
        'player_id': player_id,
        'game_id': game_id,
        'team_id': team,
        'opponent_team_id': opponent,
        'season': season,
        'week': week,
        'snaps': int(stats.get('off_snp', 0) or 0),
        'routes': 0,  # Not available in Sleeper
        'carries': int(stats.get('rush_att', 0) or 0),
        'rush_yards': float(stats.get('rush_yd', 0) or 0),
        'rush_tds': float(stats.get('rush_td', 0) or 0),
        'targets': int(stats.get('rec_tgt', 0) or 0),
        'receptions': int(stats.get('rec', 0) or 0),
        'rec_yards': float(stats.get('rec_yd', 0) or 0),
        'rec_tds': float(stats.get('rec_td', 0) or 0),
        'completions': int(stats.get('pass_cmp', 0) or 0),
        'pass_attempts': int(stats.get('pass_att', 0) or 0),
        'pass_yards': float(stats.get('pass_yd', 0) or 0),
        'pass_tds': float(stats.get('pass_td', 0) or 0),
        'interceptions': int(stats.get('pass_int', 0) or 0),
        'fumbles': int(stats.get('fum', 0) or 0),
        'fumbles_lost': int(stats.get('fum_lost', 0) or 0),
        'two_point_conversions': int(stats.get('pass_2pt', 0) or 0) + int(stats.get('rush_2pt', 0) or 0) + int(stats.get('rec_2pt', 0) or 0),
        'fantasy_points_sleeper': float(stats.get('pts_ppr', 0) or 0),
        'raw_json': json.dumps(stats),
        'ingested_at': datetime.utcnow().isoformat() + 'Z',
    }


def backfill_season(conn, season: int, players: dict, games: dict, max_weeks: int = 18):
    """Backfill all weeks for a season."""
    cursor = conn.cursor()

    # Check existing data
    cursor.execute('SELECT COUNT(*) FROM player_game_stats WHERE season = ?', (season,))
    existing = cursor.fetchone()[0]
    if existing > 0:
        print(f"  Season {season} already has {existing} rows, skipping")
        return 0

    total_inserted = 0

    for week in range(1, max_weeks + 1):
        print(f"  Fetching week {week}...", end=' ')

        stats_data = fetch_week_stats(season, week)
        if not stats_data:
            print("no data")
            continue

        week_inserted = 0
        for player_id, stats in stats_data.items():
            player_info = players.get(player_id, {})

            parsed = parse_player_stats(player_id, stats, player_info,
                                        season, week, games)
            if not parsed:
                continue

            # Check for duplicate
            cursor.execute('''
                SELECT 1 FROM player_game_stats
                WHERE player_id = ? AND game_id = ?
            ''', (parsed['player_id'], parsed['game_id']))

            if cursor.fetchone():
                continue

            # Insert
            columns = list(parsed.keys())
            placeholders = ','.join(['?' for _ in columns])
            sql = f"INSERT INTO player_game_stats ({','.join(columns)}) VALUES ({placeholders})"

            try:
                cursor.execute(sql, list(parsed.values()))
                week_inserted += 1
            except sqlite3.Error as e:
                print(f"Insert error: {e}")

        conn.commit()
        total_inserted += week_inserted
        print(f"{week_inserted} players")

        time.sleep(0.5)  # Rate limit

    return total_inserted


def main():
    print("=" * 60)
    print("Backfill Sleeper Stats for 2021-2022")
    print("=" * 60)

    conn = get_connection()

    # Load reference data
    print("\nLoading reference data...")
    players = load_players(conn)
    print(f"  {len(players)} players loaded")

    games = load_games(conn)
    print(f"  {len(games)//2} games loaded")

    # Check what seasons have games
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT season FROM games ORDER BY season')
    available_seasons = [row[0] for row in cursor.fetchall()]
    print(f"  Games available for seasons: {available_seasons}")

    # Backfill 2021 and 2022
    for season in [2021, 2022]:
        if season not in available_seasons:
            print(f"\nSeason {season}: No games in database, need to ingest games first")
            continue

        print(f"\nBackfilling season {season}...")
        count = backfill_season(conn, season, players, games)
        print(f"  Total inserted: {count}")

    # Summary
    cursor.execute('SELECT season, COUNT(*) FROM player_game_stats GROUP BY season ORDER BY season')
    print("\nFinal player_game_stats by season:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,} rows")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
