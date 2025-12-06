"""
Team Defense Stats Ingestion Script for NFL Fantasy Prediction Engine - Phase 1

STEP 6a in the ingestion pipeline as per Phase 1 v2 Section 6.6.

This script populates team_defense_game_stats with team-level defensive performance:
- Pass/rush yards allowed
- Position-specific yards/targets allowed (WR, TE, RB)
- Touchdowns allowed by position
- Efficiency metrics (EPA, success rate)
- Defensive stats (sacks, INTs, fumbles recovered)

This table is required for:
- Defensive deltas
- Baseline modeling
- Archetype matchup modeling

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.1 Fixed - Correct logger format and team nickname mapping
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    NFL_TEAMS,
    TEAM_ABBREVIATION_MAP,
    CURRENT_SEASON,
    ROLLING_WINDOW_SEASONS,
    MAX_REGULAR_SEASON_WEEKS
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Initialize logger
logger = get_ingestion_logger("ingest_stats_defense")


# Team nickname to abbreviation mapping
# Used when CSV files have team names like "Browns" instead of "CLE"
TEAM_NICKNAME_MAP = {
    "CARDINALS": "ARI", "ARIZONA CARDINALS": "ARI",
    "FALCONS": "ATL", "ATLANTA FALCONS": "ATL",
    "RAVENS": "BAL", "BALTIMORE RAVENS": "BAL",
    "BILLS": "BUF", "BUFFALO BILLS": "BUF",
    "PANTHERS": "CAR", "CAROLINA PANTHERS": "CAR",
    "BEARS": "CHI", "CHICAGO BEARS": "CHI",
    "BENGALS": "CIN", "CINCINNATI BENGALS": "CIN",
    "BROWNS": "CLE", "CLEVELAND BROWNS": "CLE",
    "COWBOYS": "DAL", "DALLAS COWBOYS": "DAL",
    "BRONCOS": "DEN", "DENVER BRONCOS": "DEN",
    "LIONS": "DET", "DETROIT LIONS": "DET",
    "PACKERS": "GB", "GREEN BAY PACKERS": "GB",
    "TEXANS": "HOU", "HOUSTON TEXANS": "HOU",
    "COLTS": "IND", "INDIANAPOLIS COLTS": "IND",
    "JAGUARS": "JAX", "JACKSONVILLE JAGUARS": "JAX",
    "CHIEFS": "KC", "KANSAS CITY CHIEFS": "KC",
    "CHARGERS": "LAC", "LOS ANGELES CHARGERS": "LAC", "LA CHARGERS": "LAC",
    "RAMS": "LAR", "LOS ANGELES RAMS": "LAR", "LA RAMS": "LAR",
    "RAIDERS": "LV", "LAS VEGAS RAIDERS": "LV",
    "DOLPHINS": "MIA", "MIAMI DOLPHINS": "MIA",
    "VIKINGS": "MIN", "MINNESOTA VIKINGS": "MIN",
    "PATRIOTS": "NE", "NEW ENGLAND PATRIOTS": "NE",
    "SAINTS": "NO", "NEW ORLEANS SAINTS": "NO",
    "GIANTS": "NYG", "NEW YORK GIANTS": "NYG", "NY GIANTS": "NYG",
    "JETS": "NYJ", "NEW YORK JETS": "NYJ", "NY JETS": "NYJ",
    "EAGLES": "PHI", "PHILADELPHIA EAGLES": "PHI",
    "STEELERS": "PIT", "PITTSBURGH STEELERS": "PIT",
    "49ERS": "SF", "SAN FRANCISCO 49ERS": "SF", "NINERS": "SF",
    "SEAHAWKS": "SEA", "SEATTLE SEAHAWKS": "SEA",
    "BUCCANEERS": "TB", "TAMPA BAY BUCCANEERS": "TB", "BUCS": "TB",
    "TITANS": "TEN", "TENNESSEE TITANS": "TEN",
    "COMMANDERS": "WAS", "WASHINGTON COMMANDERS": "WAS",
}


class TeamDefenseStatsIngestion:
    """
    Handles ingestion of team-level defensive statistics.
    
    Since Sleeper doesn't provide team defense stats directly, we aggregate
    from offensive stats - what the offense did against a defense tells us
    how that defense performed.
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize team defense stats ingestion.
        
        Args:
            db: Optional database connection. Uses default if not provided.
        """
        self.db = db or get_db()
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "errors": 0
        }
    
    def normalize_team_id(self, team_id: str) -> Optional[str]:
        """
        Normalize team abbreviation.
        
        Handles:
        - Standard abbreviations (CLE, DET, etc.)
        - Team nicknames (Browns, Lions, etc.)
        - Full team names (Cleveland Browns, Detroit Lions, etc.)
        """
        if not team_id:
            return None
        
        team_id_upper = str(team_id).upper().strip()
        
        # First check if it's a nickname
        if team_id_upper in TEAM_NICKNAME_MAP:
            return TEAM_NICKNAME_MAP[team_id_upper]
        
        # Then check standard abbreviation map
        return TEAM_ABBREVIATION_MAP.get(team_id_upper, None)

    # ------------------------------------------------------------------
    # Season-level team coverage style from FTN Team Stats CSV
    # ------------------------------------------------------------------
    def load_team_season_coverage_from_csv(self, csv_path: Path, season: int) -> int:
        """
        Ingest season-level team coverage style stats (man/zone, MOFC/MOFO)
        from a CSV like '2025 Team Stats.csv'.
        
        CSV format expected:
        Team,Man Rate,Zone Rate,Middle Closed Rate,Middle Open Rate
        Browns,42.1,53.9,61.5,37.6
        ...
        """
        import pandas as pd

        if not csv_path.exists():
            logger.error(f"Team season coverage CSV not found: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        inserted = 0
        updated = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for _, row in df.iterrows():
                team_abbr = str(row.get("Team") or "").strip()
                if not team_abbr:
                    continue

                team_id = self.normalize_team_id(team_abbr)
                if not team_id:
                    # If we can't normalize the abbreviation, skip this row
                    logger.warning(
                        f"Skipping team season coverage row for unknown team abbreviation '{team_abbr}'"
                    )
                    continue

                # Ensure this team_id actually exists in the teams table
                cursor.execute(
                    "SELECT 1 FROM teams WHERE team_id = ?",
                    (team_id,),
                )
                if cursor.fetchone() is None:
                    logger.warning(
                        f"Skipping team season coverage row for unknown team_id '{team_id}' (abbr '{team_abbr}')"
                    )
                    continue

                # Only now do we parse the numeric fields
                man_rate = float(row.get("Man Rate") or 0.0)
                zone_rate = float(row.get("Zone Rate") or 0.0)
                middle_closed_rate = float(row.get("Middle Closed Rate") or 0.0)
                middle_open_rate = float(row.get("Middle Open Rate") or 0.0)

                # Check if we already have a row for this team/season
                cursor.execute(
                    """
                    SELECT 1
                      FROM team_defense_season_stats
                     WHERE team_id = ? AND season = ?
                    """,
                    (team_id, season),
                )
                exists = cursor.fetchone() is not None

                if exists:
                    # Update existing row
                    cursor.execute(
                        """
                        UPDATE team_defense_season_stats
                           SET man_rate = ?,
                               zone_rate = ?,
                               middle_closed_rate = ?,
                               middle_open_rate = ?,
                               updated_at = CURRENT_TIMESTAMP
                         WHERE team_id = ? AND season = ?
                        """,
                        (
                            man_rate,
                            zone_rate,
                            middle_closed_rate,
                            middle_open_rate,
                            team_id,
                            season,
                        ),
                    )
                    updated += 1
                else:
                    # Insert new row
                    cursor.execute(
                        """
                        INSERT INTO team_defense_season_stats (
                            team_id,
                            season,
                            man_rate,
                            zone_rate,
                            middle_closed_rate,
                            middle_open_rate,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """,
                        (
                            team_id,
                            season,
                            man_rate,
                            zone_rate,
                            middle_closed_rate,
                            middle_open_rate,
                        ),
                    )
                    inserted += 1

        logger.info(
            f"Team defense season coverage ingestion for {season}: {inserted} inserted, {updated} updated"
        )
        return inserted + updated
    
    def generate_game_id(self, season: int, week: int, 
                         team: str, opponent: str) -> str:
        """Generate game ID matching the format from ingest_games."""
        team1, team2 = sorted([team, opponent])
        return f"{season}_{week:02d}_{team1}_{team2}"
    
    def aggregate_defense_stats_from_offense(self, season: int, week: int) -> Dict[str, Dict]:
        """
        Aggregate defensive stats by analyzing what offenses did against each defense.
        
        For each defensive team, we sum up what all opposing offensive players
        accumulated against them.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            Dictionary of team_id -> aggregated defense stats
        """
        logger.info(f"Aggregating defense stats for {season} week {week}...")
        
        # Query all offensive stats for this week
        query = """
            SELECT 
                pgs.opponent_team_id as defense_team,
                pgs.team_id as offense_team,
                pgs.game_id,
                p.position,
                pgs.targets,
                pgs.receptions,
                pgs.rec_yards,
                pgs.rec_tds,
                pgs.carries,
                pgs.rush_yards,
                pgs.rush_tds,
                pgs.pass_yards,
                pgs.pass_tds,
                pgs.interceptions,
                pgs.fumbles_lost
            FROM player_game_stats pgs
            JOIN players p ON pgs.player_id = p.player_id
            WHERE pgs.season = ? AND pgs.week = ?
        """
        
        rows = self.db.fetch_all(query, (season, week))
        
        # Aggregate by defensive team
        defense_stats = defaultdict(lambda: {
            'game_id': None,
            'opponent_team_id': None,
            'total_yards_allowed': 0,
            'pass_yards_allowed': 0,
            'rush_yards_allowed': 0,
            'total_tds_allowed': 0,
            'pass_tds_allowed': 0,
            'rush_tds_allowed': 0,
            'wr_targets_allowed': 0,
            'wr_yards_allowed': 0,
            'wr_tds_allowed': 0,
            'te_targets_allowed': 0,
            'te_yards_allowed': 0,
            'te_tds_allowed': 0,
            'rb_targets_allowed': 0,
            'rb_yards_allowed': 0,
            'rb_rec_tds_allowed': 0,
            'rb_rush_yards_allowed': 0,
            'rb_rush_tds_allowed': 0,
            'sacks': 0,
            'interceptions': 0,
            'fumbles_recovered': 0,
        })
        
        for row in rows:
            defense_team = row['defense_team']
            if not defense_team:
                continue
            
            stats = defense_stats[defense_team]
            stats['game_id'] = row['game_id']
            stats['opponent_team_id'] = row['offense_team']
            
            position = row['position']
            
            # Aggregate receiving stats
            targets = row['targets'] or 0
            rec_yards = row['rec_yards'] or 0
            rec_tds = row['rec_tds'] or 0
            
            stats['pass_yards_allowed'] += rec_yards
            stats['total_yards_allowed'] += rec_yards
            stats['pass_tds_allowed'] += rec_tds
            stats['total_tds_allowed'] += rec_tds
            
            if position == 'WR':
                stats['wr_targets_allowed'] += targets
                stats['wr_yards_allowed'] += rec_yards
                stats['wr_tds_allowed'] += rec_tds
            elif position == 'TE':
                stats['te_targets_allowed'] += targets
                stats['te_yards_allowed'] += rec_yards
                stats['te_tds_allowed'] += rec_tds
            elif position == 'RB':
                stats['rb_targets_allowed'] += targets
                stats['rb_yards_allowed'] += rec_yards
                stats['rb_rec_tds_allowed'] += rec_tds
            
            # Aggregate rushing stats
            rush_yards = row['rush_yards'] or 0
            rush_tds = row['rush_tds'] or 0
            
            stats['rush_yards_allowed'] += rush_yards
            stats['total_yards_allowed'] += rush_yards
            stats['rush_tds_allowed'] += rush_tds
            stats['total_tds_allowed'] += rush_tds
            
            if position == 'RB':
                stats['rb_rush_yards_allowed'] += rush_yards
                stats['rb_rush_tds_allowed'] += rush_tds
            
            # Aggregate turnovers forced
            stats['interceptions'] += row['interceptions'] or 0
            stats['fumbles_recovered'] += row['fumbles_lost'] or 0
        
        return dict(defense_stats)
    
    def upsert_defense_game_stats(self, team_id: str, season: int, week: int,
                                   stats: Dict, conn) -> bool:
        """Insert or update team defense game stats."""
        try:
            cursor = conn.cursor()
            
            game_id = stats.get('game_id') or self.generate_game_id(
                season, week, team_id, stats.get('opponent_team_id', 'UNK')
            )
            
            # Check if record exists
            cursor.execute(
                """SELECT 1 FROM team_defense_game_stats 
                   WHERE team_id = ? AND season = ? AND week = ?""",
                (team_id, season, week)
            )
            exists = cursor.fetchone() is not None
            
            if exists:
                cursor.execute("""
                    UPDATE team_defense_game_stats SET
                        game_id = ?,
                        opponent_team_id = ?,
                        total_yards_allowed = ?,
                        pass_yards_allowed = ?,
                        rush_yards_allowed = ?,
                        total_tds_allowed = ?,
                        pass_tds_allowed = ?,
                        rush_tds_allowed = ?,
                        wr_targets_allowed = ?,
                        wr_yards_allowed = ?,
                        wr_tds_allowed = ?,
                        te_targets_allowed = ?,
                        te_yards_allowed = ?,
                        te_tds_allowed = ?,
                        rb_targets_allowed = ?,
                        rb_yards_allowed = ?,
                        rb_rec_tds_allowed = ?,
                        rb_rush_yards_allowed = ?,
                        rb_rush_tds_allowed = ?,
                        sacks = ?,
                        interceptions = ?,
                        fumbles_recovered = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE team_id = ? AND season = ? AND week = ?
                """, (
                    game_id,
                    stats.get('opponent_team_id'),
                    stats.get('total_yards_allowed', 0),
                    stats.get('pass_yards_allowed', 0),
                    stats.get('rush_yards_allowed', 0),
                    stats.get('total_tds_allowed', 0),
                    stats.get('pass_tds_allowed', 0),
                    stats.get('rush_tds_allowed', 0),
                    stats.get('wr_targets_allowed', 0),
                    stats.get('wr_yards_allowed', 0),
                    stats.get('wr_tds_allowed', 0),
                    stats.get('te_targets_allowed', 0),
                    stats.get('te_yards_allowed', 0),
                    stats.get('te_tds_allowed', 0),
                    stats.get('rb_targets_allowed', 0),
                    stats.get('rb_yards_allowed', 0),
                    stats.get('rb_rec_tds_allowed', 0),
                    stats.get('rb_rush_yards_allowed', 0),
                    stats.get('rb_rush_tds_allowed', 0),
                    stats.get('sacks', 0),
                    stats.get('interceptions', 0),
                    stats.get('fumbles_recovered', 0),
                    team_id, season, week
                ))
                self.stats["updated"] += 1
            else:
                cursor.execute("""
                    INSERT INTO team_defense_game_stats (
                        team_id, season, week, game_id, opponent_team_id,
                        total_yards_allowed, pass_yards_allowed, rush_yards_allowed,
                        total_tds_allowed, pass_tds_allowed, rush_tds_allowed,
                        wr_targets_allowed, wr_yards_allowed, wr_tds_allowed,
                        te_targets_allowed, te_yards_allowed, te_tds_allowed,
                        rb_targets_allowed, rb_yards_allowed, rb_rec_tds_allowed,
                        rb_rush_yards_allowed, rb_rush_tds_allowed,
                        sacks, interceptions, fumbles_recovered,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                              CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    team_id, season, week, game_id, stats.get('opponent_team_id'),
                    stats.get('total_yards_allowed', 0),
                    stats.get('pass_yards_allowed', 0),
                    stats.get('rush_yards_allowed', 0),
                    stats.get('total_tds_allowed', 0),
                    stats.get('pass_tds_allowed', 0),
                    stats.get('rush_tds_allowed', 0),
                    stats.get('wr_targets_allowed', 0),
                    stats.get('wr_yards_allowed', 0),
                    stats.get('wr_tds_allowed', 0),
                    stats.get('te_targets_allowed', 0),
                    stats.get('te_yards_allowed', 0),
                    stats.get('te_tds_allowed', 0),
                    stats.get('rb_targets_allowed', 0),
                    stats.get('rb_yards_allowed', 0),
                    stats.get('rb_rec_tds_allowed', 0),
                    stats.get('rb_rush_yards_allowed', 0),
                    stats.get('rb_rush_tds_allowed', 0),
                    stats.get('sacks', 0),
                    stats.get('interceptions', 0),
                    stats.get('fumbles_recovered', 0)
                ))
                self.stats["inserted"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error upserting defense stats for {team_id}: {e}")
            self.stats["errors"] += 1
            return False
    
    def ingest_week(self, season: int, week: int) -> Dict:
        """Ingest team defense stats for a specific week."""
        logger.info(f"Ingesting team defense stats for {season} week {week}...")
        
        # Aggregate from offensive stats
        defense_stats = self.aggregate_defense_stats_from_offense(season, week)
        
        if not defense_stats:
            logger.warning(f"No defense stats found for {season} week {week}")
            return self.stats.copy()
        
        # Upsert each team's stats
        with self.db.get_connection() as conn:
            for team_id, stats in defense_stats.items():
                self.upsert_defense_game_stats(team_id, season, week, stats, conn)
            conn.commit()
        
        logger.info(
            f"Team defense stats for {season} week {week}: "
            f"{self.stats['inserted']} inserted, {self.stats['updated']} updated"
        )
        
        return self.stats.copy()
    
    def ingest_season(self, season: int) -> Dict:
        """Ingest team defense stats for an entire season."""
        logger.info(f"Ingesting team defense stats for season {season}...")
        
        for week in range(1, MAX_REGULAR_SEASON_WEEKS + 1):
            self.ingest_week(season, week)
        
        return self.stats.copy()
    
    def run(self, seasons: Optional[List[int]] = None) -> Dict:
        """
        Run team defense stats ingestion for specified seasons.
        
        Args:
            seasons: List of seasons to process. Defaults to rolling window.
        
        Returns:
            Final statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING TEAM DEFENSE STATS INGESTION")
        logger.info("=" * 60)
        
        if seasons is None:
            seasons = [CURRENT_SEASON - i for i in range(ROLLING_WINDOW_SEASONS)]
        
        start_time = time.time()
        
        for season in seasons:
            self.ingest_season(season)
        
        elapsed = time.time() - start_time
        
        total = self.stats["inserted"] + self.stats["updated"]
        logger.info(f"Ingested {total} rows in {elapsed:.2f}s")
        
        logger.info("=" * 60)
        logger.info("TEAM DEFENSE STATS INGESTION COMPLETE")
        logger.info("=" * 60)
        
        return {
            **self.stats,
            "elapsed_seconds": elapsed
        }


def run_team_defense_ingestion(seasons: Optional[List[int]] = None,
                                db: Optional[DatabaseConnection] = None) -> Dict:
    """
    Convenience function to run team defense stats ingestion.
    """
    ingestion = TeamDefenseStatsIngestion(db=db)
    return ingestion.run(seasons=seasons)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest team defense stats")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        help="Seasons to ingest"
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Specific week to ingest (current season only)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help="Path to team stats CSV for season-level coverage data"
    )
    parser.add_argument(
        "--csv-season",
        type=int,
        default=CURRENT_SEASON,
        help="Season for the CSV data"
    )
    
    args = parser.parse_args()
    
    ingestion = TeamDefenseStatsIngestion()
    
    if args.csv:
        csv_path = Path(args.csv)
        count = ingestion.load_team_season_coverage_from_csv(csv_path, args.csv_season)
        print(f"Loaded {count} team season coverage records from {csv_path}")
    elif args.week:
        result = ingestion.ingest_week(CURRENT_SEASON, args.week)
        print(f"Week {args.week}: {result['inserted']} inserted, {result['updated']} updated")
    else:
        result = ingestion.run(seasons=args.seasons)
        print(f"\nTeam Defense Stats Ingestion Complete:")
        print(f"  Inserted: {result['inserted']}")
        print(f"  Updated: {result['updated']}")
        print(f"  Errors: {result['errors']}")
