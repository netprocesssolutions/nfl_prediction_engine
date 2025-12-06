"""
Individual Defender Stats Ingestion Script for NFL Fantasy Prediction Engine - Phase 1

STEP 6b in the ingestion pipeline as per Phase 1 v2 Section 6.6.

This script populates defender_game_stats with individual defender performance:
- Coverage snaps
- Targets allowed
- Receptions allowed
- Yards allowed (total and YAC)
- TDs allowed
- Alignment percentages (boundary, slot, deep, box)
- Coverage type percentages (man vs zone)

This table powers:
- Defender-aware model in Phase 4
- Archetype modeling
- Coverage probability modeling

Note: Detailed defender stats require nflfastR or similar data source.
This script provides a framework that can be populated when data is available.

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from config.settings import (
    DEFENSIVE_POSITIONS,
    DEFENSIVE_POSITION_GROUPS,
    TEAM_ABBREVIATION_MAP,
    SLEEPER_ENDPOINTS,
    API_TIMEOUT,
    API_RETRY_ATTEMPTS,
    API_RETRY_DELAY,
    CURRENT_SEASON,
    ROLLING_WINDOW_SEASONS,
    MAX_REGULAR_SEASON_WEEKS,
)
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Initialize logger
logger = get_ingestion_logger("ingest_stats_defenders")


class DefenderStatsIngestion:
    """
    Handles ingestion of individual defender game statistics.
    
    This is a v2 requirement for defender-aware modeling.
    
    Data sources:
    - nflfastR (primary - contains play-by-play with coverage data)
    - PFF data (if available via manual CSV)
    - Sleeper API (limited defender data)
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize defender stats ingestion.
        
        Args:
            db: Optional database connection.
        """
        self.db = db or get_db()
        self.stats = {
            "inserted": 0,
            "updated": 0,
            "skipped": 0,
            "errors": 0
        }
        
        # Cache defenders and (season, week, team) â†’ game_id lookups
        self._defenders: Dict[str, Dict] = {}
        self._team_week_game_cache: Dict[Tuple[int, int, str], Optional[str]] = {}
        self._load_defenders()
    
    def _load_defenders(self):
        """Load defenders from database."""
        logger.info("Loading defenders...")
        
        results = self.db.fetch_all(
            "SELECT defender_id, team_id, position_group FROM defenders"
        )
        
        self._defenders = {
            row["defender_id"]: {
                "team_id": row["team_id"],
                "position_group": row["position_group"]
            }
            for row in results
        }
        
        logger.info(f"Loaded {len(self._defenders)} defenders")
    
    def normalize_team_id(self, team_id: str) -> Optional[str]:
        """Normalize team abbreviation."""
        if not team_id:
            return None
        team_id = team_id.upper().strip()
        return TEAM_ABBREVIATION_MAP.get(team_id, team_id)
    
    def get_game_for_team(self, season: int, week: int, team_id: Optional[str]) -> Optional[str]:
        """
        Look up this team's game_id for a given season/week using the games table.
        
        Returns:
            game_id string, or None if no game is found.
        """
        if not team_id:
            return None

        # Normalize the team ID to match teams table
        team_id = self.normalize_team_id(team_id)
        if not team_id:
            return None

        key = (season, week, team_id)
        if key in self._team_week_game_cache:
            return self._team_week_game_cache[key]

        row = self.db.fetch_one(
            """
            SELECT game_id
            FROM games
            WHERE season = ? AND week = ?
              AND (home_team_id = ? OR away_team_id = ?)
            """,
            (season, week, team_id, team_id),
        )

        if not row:
            logger.warning(
                f"No game found for team {team_id} in {season} week {week}"
            )
            self._team_week_game_cache[key] = None
            return None

        game_id = row["game_id"]
        self._team_week_game_cache[key] = game_id
        return game_id
    
    def create_placeholder_stats(self, defender_id: str, game_id: str,
                                  season: int, week: int) -> Dict:
        """
        Create placeholder defender stats for a game.
        
        When detailed coverage data isn't available, we create
        placeholder records that can be updated later.
        
        Args:
            defender_id: Defender ID
            game_id: Game ID
            season: Season year
            week: Week number
        
        Returns:
            Placeholder stats dictionary
        """
        return {
            "defender_id": defender_id,
            "game_id": game_id,
            "season": season,
            "week": week,
            "snaps": 0,
            "coverage_snaps": 0,
            "targets_allowed": 0,
            "receptions_allowed": 0,
            "yards_allowed": 0.0,
            "yac_allowed": 0.0,
            "ypr_allowed": 0.0,
            "tds_allowed": 0.0,
            "alignment_boundary_pct": 0.0,
            "alignment_slot_pct": 0.0,
            "alignment_deep_pct": 0.0,
            "alignment_box_pct": 0.0,
            "man_coverage_pct": 0.0,
            "zone_coverage_pct": 0.0,
            "penalties": 0,
            "pass_breakups": 0,
            "interceptions": 0,
            "raw_json": None
        }

    # ------------------------------------------------------------------
    # Season-level coverage ingestion from FTN Player Stats CSVs
    # ------------------------------------------------------------------
    def _parse_season_coverage_row(self, row: dict, season: int) -> Optional[Dict[str, Any]]:
        """
        Map a single FTN '20xx Player Stats.csv' row into defender_season_coverage_stats.
        """
        name = str(row.get("Pass Coverage Player") or "").strip()
        if not name:
            return None

        # Find defender_id by full_name match (basic version; can be improved later).
        defender = self.db.fetch_one(
            "SELECT defender_id, team_id FROM defenders WHERE full_name = ?",
            (name,)
        )
        if not defender:
            # Could log here if you want visibility
            return None

        defender_id = defender["defender_id"]
        team_id = defender["team_id"]

        def f(col, default=0.0):
            v = row.get(col)
            if v in (None, "", " "):
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def i(col, default=0):
            return int(round(f(col, default)))

        games = i("Pass Coverage GAME")
        games_started = i("Pass Coverage GameStart")

        targets = i("Pass Coverage Tgt")
        completions = i("Pass Coverage Cmp")
        interceptions = i("Pass Coverage Int")
        completion_pct = f("Pass Coverage Cmp%")
        yards_allowed = f("Pass Coverage Yds")
        yards_per_completion = f("Pass Coverage Yds/Cmp")
        rating_against = f("Pass Rush Rat")  # this column appears to be rating vs

        air_yards_allowed = f("Pass Rush Air")
        yac_allowed = f("Pass Rush YAC")

        blitzes = i("Tackles Bltz")
        hurries = i("Tackles Hrry")
        qb_hits = i("Tackles QBKD")
        passes_defensed = i("additional Bats")
        sacks = f(" Sk")
        pressures = f(" Prss")
        total_tackles = f(" Comb")
        missed_tackles = f(" MTkl")
        missed_tackle_pct = f(" MTkl%")

        return {
            "defender_id": defender_id,
            "season": season,
            "team_id": team_id,
            "games": games,
            "games_started": games_started,
            "targets": targets,
            "completions": completions,
            "completion_pct": completion_pct,
            "interceptions": interceptions,
            "yards_allowed": yards_allowed,
            "yards_per_completion": yards_per_completion,
            "rating_against": rating_against,
            "air_yards_allowed": air_yards_allowed,
            "yac_allowed": yac_allowed,
            "blitzes": blitzes,
            "hurries": hurries,
            "qb_hits": qb_hits,
            "passes_defensed": passes_defensed,
            "sacks": sacks,
            "total_tackles": total_tackles,
            "missed_tackles": missed_tackles,
            "missed_tackle_pct": missed_tackle_pct,
            "raw_json": json.dumps(row),
        }

    def load_season_coverage_from_csv(self, csv_path: Path, season: int) -> int:
        """
        Ingest season-level defender coverage stats from an FTN Player Stats CSV.
        """
        import pandas as pd

        if not csv_path.exists():
            logger.error(f"Defender season coverage CSV not found: {csv_path}")
            return 0

        df = pd.read_csv(csv_path)
        inserted = 0
        updated = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for _, row in df.iterrows():
                payload = self._parse_season_coverage_row(row.to_dict(), season)
                if not payload:
                    continue

                # Upsert logic
                cursor.execute(
                    """
                    SELECT 1 FROM defender_season_coverage_stats
                    WHERE defender_id = ? AND season = ?
                    """,
                    (payload["defender_id"], payload["season"])
                )
                exists = cursor.fetchone() is not None

                if exists:
                    cursor.execute(
                        """
                        UPDATE defender_season_coverage_stats
                        SET team_id = ?,
                            games = ?,
                            games_started = ?,
                            targets = ?,
                            completions = ?,
                            completion_pct = ?,
                            interceptions = ?,
                            yards_allowed = ?,
                            yards_per_completion = ?,
                            rating_against = ?,
                            air_yards_allowed = ?,
                            yac_allowed = ?,
                            blitzes = ?,
                            hurries = ?,
                            qb_hits = ?,
                            passes_defensed = ?,
                            sacks = ?,
                            total_tackles = ?,
                            missed_tackles = ?,
                            missed_tackle_pct = ?,
                            raw_json = ?
                        WHERE defender_id = ? AND season = ?
                        """,
                        (
                            payload["team_id"],
                            payload["games"],
                            payload["games_started"],
                            payload["targets"],
                            payload["completions"],
                            payload["completion_pct"],
                            payload["interceptions"],
                            payload["yards_allowed"],
                            payload["yards_per_completion"],
                            payload["rating_against"],
                            payload["air_yards_allowed"],
                            payload["yac_allowed"],
                            payload["blitzes"],
                            payload["hurries"],
                            payload["qb_hits"],
                            payload["passes_defensed"],
                            payload["sacks"],
                            payload["total_tackles"],
                            payload["missed_tackles"],
                            payload["missed_tackle_pct"],
                            payload["raw_json"],
                            payload["defender_id"],
                            payload["season"],
                        ),
                    )
                    updated += 1
                else:
                    cursor.execute(
                        """
                        INSERT INTO defender_season_coverage_stats (
                            defender_id, season, team_id,
                            games, games_started,
                            targets, completions, completion_pct,
                            interceptions, yards_allowed, yards_per_completion,
                            rating_against, air_yards_allowed, yac_allowed,
                            blitzes, hurries, qb_hits,
                            passes_defensed, sacks, total_tackles,
                            missed_tackles, missed_tackle_pct, raw_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload["defender_id"],
                            payload["season"],
                            payload["team_id"],
                            payload["games"],
                            payload["games_started"],
                            payload["targets"],
                            payload["completions"],
                            payload["completion_pct"],
                            payload["interceptions"],
                            payload["yards_allowed"],
                            payload["yards_per_completion"],
                            payload["rating_against"],
                            payload["air_yards_allowed"],
                            payload["yac_allowed"],
                            payload["blitzes"],
                            payload["hurries"],
                            payload["qb_hits"],
                            payload["passes_defensed"],
                            payload["sacks"],
                            payload["total_tackles"],
                            payload["missed_tackles"],
                            payload["missed_tackle_pct"],
                            payload["raw_json"],
                        ),
                    )
                    inserted += 1

        logger.info(
            f"Defender season coverage ingestion for {season}: "
            f"{inserted} inserted, {updated} updated"
        )
        return inserted + updated
    
    def parse_nflfastr_defender_stats(self, row: Dict) -> Optional[Dict]:
        """
        Parse defender stats from nflfastR play-by-play data.
        
        This would be called when processing nflfastR CSV/parquet files.
        
        Args:
            row: Row from nflfastR data
        
        Returns:
            Parsed stats dictionary or None
        """
        # This is a placeholder - actual implementation depends on
        # nflfastR column names and data structure
        
        # nflfastR typically has columns like:
        # - defteam (defensive team)
        # - yards_gained
        # - air_yards
        # - yards_after_catch
        # - pass_defense_1_player_id, pass_defense_2_player_id
        # - coverage_type (man/zone)
        
        # For now, return None - will be implemented when nflfastR
        # data is integrated
        return None
    
    def parse_stat_value(self, value, stat_type: str = "int"):
        """
        Safely parse numeric/stat values coming from Sleeper.

        Args:
            value: Raw value from API
            stat_type: "int", "real", or "str"

        Returns:
            Parsed value with sensible defaults
        """
        try:
            if value is None or value == "":
                return 0 if stat_type in ["int", "real"] else None

            if stat_type == "int":
                return int(float(value))
            elif stat_type == "real":
                return float(value)
            return value
        except (ValueError, TypeError):
            return 0 if stat_type in ["int", "real"] else None

    def fetch_week_stats(self, season: int, week: int) -> Optional[Dict[str, Dict]]:
        """
        Fetch per-player stats for a specific week from Sleeper.

        We reuse the same stats_regular endpoint used by the offensive ingestion.

        Args:
            season: Season year
            week: Week number

        Returns:
            Dict mapping player_id -> raw stats dict, or None on error
        """
        url = SLEEPER_ENDPOINTS["stats_regular"](season, week)

        for attempt in range(API_RETRY_ATTEMPTS):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=API_TIMEOUT)
                duration = time.time() - start_time
                logger.info(
                    f"Sleeper stats_regular {season} week {week} "
                    f"responded in {duration:.2f}s (status {response.status_code})"
                )

                if response.status_code != 200:
                    logger.warning(
                        f"Failed to fetch defender stats for {season} week {week} "
                        f"(status {response.status_code}), attempt {attempt + 1}"
                    )
                    time.sleep(API_RETRY_DELAY)
                    continue

                data = response.json() or []
                if not isinstance(data, list):
                    logger.error(
                        f"Unexpected payload type from Sleeper stats_regular: "
                        f"{type(data)}"
                    )
                    return None

                stats_by_player: Dict[str, Dict] = {}
                for entry in data:
                    player_id = entry.get("player_id")
                    if not player_id:
                        continue
                    stats_by_player[str(player_id)] = entry

                logger.info(
                    f"Fetched {len(stats_by_player)} total player stat entries "
                    f"for {season} week {week}"
                )
                return stats_by_player

            except requests.RequestException as e:
                logger.warning(
                    f"Error fetching defender stats for {season} week {week} "
                    f"(attempt {attempt + 1}): {e}"
                )
                time.sleep(API_RETRY_DELAY)

        logger.error(f"Giving up on defender stats for {season} week {week}")
        return None
    
    def extract_defender_stats(
        self, defender_id: str, stats: Dict, season: int, week: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract and format stats for a single defensive player from Sleeper.

        Sleeper does not expose detailed coverage metrics, so for now we:
        - Populate interceptions and pass breakups from available IDP fields
        - Keep coverage-related fields at 0 as placeholders
        """
        if not isinstance(stats, dict):
            return None

        defender_info = self._defenders.get(defender_id)
        if not defender_info:
            self.stats["skipped"] += 1
            return None

        team_id = defender_info.get("team_id")
        if not team_id:
            self.stats["skipped"] += 1
            return None

        game_id = self.get_game_for_team(season, week, team_id)
        if not game_id:
            self.stats["skipped"] += 1
            return None

        # Basic playing time proxy (if available)
        snaps = self.parse_stat_value(
            stats.get("def_snaps") or stats.get("snaps"), "int"
        )

        # IDP-style counting stats
        interceptions = self.parse_stat_value(
            stats.get("int") or stats.get("ints"), "int"
        )
        pass_breakups = self.parse_stat_value(
            stats.get("pd") or stats.get("pass_defended"), "int"
        )
        penalties = self.parse_stat_value(
            stats.get("pen") or stats.get("penalties"), "int"
        )

        return {
            "defender_id": defender_id,
            "game_id": game_id,
            "season": season,
            "week": week,
            # Playing time
            "snaps": snaps,
            "coverage_snaps": 0,
            # Coverage results (not available from Sleeper yet)
            "targets_allowed": 0,
            "receptions_allowed": 0,
            "yards_allowed": 0.0,
            "yac_allowed": 0.0,
            "ypr_allowed": 0.0,
            "tds_allowed": 0.0,
            # Alignment placeholders
            "alignment_boundary_pct": 0.0,
            "alignment_slot_pct": 0.0,
            "alignment_deep_pct": 0.0,
            "alignment_box_pct": 0.0,
            # Coverage type placeholders
            "man_coverage_pct": 0.0,
            "zone_coverage_pct": 0.0,
            # IDP impact stats
            "penalties": penalties,
            "pass_breakups": pass_breakups,
            "interceptions": interceptions,
            # Raw JSON dump for reproducibility
            "raw_json": json.dumps(stats),
        }

    def ingest_week(self, season: int, week: int) -> int:
        """
        Ingest all defender stats for a specific week from Sleeper.
        """
        logger.info(f"Ingesting defender stats for {season} week {week}...")
        stats_data = self.fetch_week_stats(season, week)

        if not stats_data:
            logger.info(f"No defender stats data for {season} week {week}")
            return 0

        processed = 0

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for player_id, player_stats in stats_data.items():
                defender_id = str(player_id)

                # Only process players that exist in the defenders table
                if defender_id not in self._defenders:
                    continue

                try:
                    formatted = self.extract_defender_stats(
                        defender_id, player_stats, season, week
                    )
                    if not formatted:
                        continue

                    if self.insert_defender_stats(formatted, cursor):
                        processed += 1
                except Exception as e:
                    logger.error(
                        f"Error processing defender {defender_id} for "
                        f"{season} week {week}: {e}"
                    )
                    self.stats["errors"] += 1

        logger.info(
            f"Finished defender stats for {season} week {week}: "
            f"{processed} records processed"
        )
        return processed

    def load_from_csv(self, filepath: Path, season: int) -> int:
        """
        Load defender stats from a CSV file.
        
        Expected columns:
        - defender_id or player_id
        - game_id or (season, week, team, opponent)
        - snaps, coverage_snaps
        - targets_allowed, receptions_allowed, yards_allowed
        - man_coverage_pct, zone_coverage_pct
        - alignment columns
        
        Args:
            filepath: Path to CSV file
            season: Season year (for validation)
        
        Returns:
            Number of records loaded
        """
        logger.info(f"Loading defender stats from {filepath}...")
        
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            loaded = 0
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    try:
                        # Map columns (adjust based on actual CSV structure)
                        defender_id = str(row.get("defender_id") or row.get("player_id", ""))
                        
                        if not defender_id or defender_id not in self._defenders:
                            self.stats["skipped"] += 1
                            continue
                        
                        defender_info = self._defenders[defender_id]
                        team_id = defender_info.get("team_id")

                        # Determine season/week for this row (fallback to function argument)
                        row_season = int(row.get("season", season))
                        row_week = int(row.get("week", 0))

                        if row_week <= 0:
                            logger.warning(
                                f"Skipping defender {defender_id}: missing/invalid week in CSV "
                                f"(season={row_season}, raw_week={row.get('week')})"
                            )
                            self.stats["skipped"] += 1
                            continue

                        # Resolve game_id from the games table (canonical source of truth)
                        game_id = self.get_game_for_team(row_season, row_week, team_id)
                        if not game_id:
                            logger.warning(
                                f"Skipping defender {defender_id}: no game found for team "
                                f"{team_id} in {row_season} week {row_week}"
                            )
                            self.stats["skipped"] += 1
                            continue

                        # Optional: if CSV provided a game_id and it disagrees, log but still use canonical
                        csv_game_id = str(row.get("game_id") or "").strip()
                        if csv_game_id and csv_game_id != game_id:
                            logger.warning(
                                f"CSV game_id {csv_game_id} for defender {defender_id} "
                                f"does not match canonical game_id {game_id}; using canonical value."
                            )

                        stats = {
                            "defender_id": defender_id,
                            "game_id": game_id,
                            "season": row_season,
                            "week": row_week,
                            "snaps": int(row.get("snaps", 0)),
                            "coverage_snaps": int(row.get("coverage_snaps", 0)),
                            "targets_allowed": int(row.get("targets_allowed", 0)),
                            "receptions_allowed": int(row.get("receptions_allowed", 0)),
                            "yards_allowed": float(row.get("yards_allowed", 0)),
                            "yac_allowed": float(row.get("yac_allowed", 0)),
                            "ypr_allowed": float(row.get("ypr_allowed", 0)),
                            "tds_allowed": float(row.get("tds_allowed", 0)),
                            "alignment_boundary_pct": float(row.get("alignment_boundary_pct", 0)),
                            "alignment_slot_pct": float(row.get("alignment_slot_pct", 0)),
                            "alignment_deep_pct": float(row.get("alignment_deep_pct", 0)),
                            "alignment_box_pct": float(row.get("alignment_box_pct", 0)),
                            "man_coverage_pct": float(row.get("man_coverage_pct", 0)),
                            "zone_coverage_pct": float(row.get("zone_coverage_pct", 0)),
                            "penalties": int(row.get("penalties", 0)),
                            "pass_breakups": int(row.get("pass_breakups", 0)),
                            "interceptions": int(row.get("interceptions", 0)),
                            "raw_json": json.dumps(row.to_dict())
                        }

                        if self.insert_defender_stats(stats, cursor):
                            loaded += 1
                            
                    except Exception as e:
                        logger.error(f"Error processing CSV row: {e}")
                        self.stats["errors"] += 1
            
            logger.info(f"Loaded {loaded} defender stat records from CSV")
            return loaded
            
        except ImportError:
            logger.warning("pandas not available for CSV loading")
            return 0
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return 0
    
    def insert_defender_stats(self, stats: Dict, cursor) -> bool:
        """
        Insert or update defender stats.
        
        Args:
            stats: Stats dictionary
            cursor: Database cursor
        
        Returns:
            True if successful
        """
        try:
            # Check for existing
            cursor.execute(
                "SELECT defender_id FROM defender_game_stats WHERE defender_id = ? AND game_id = ?",
                (stats["defender_id"], stats["game_id"])
            )
            existing = cursor.fetchone()
            
            timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            
            if existing:
                cursor.execute("""
                    UPDATE defender_game_stats SET
                        snaps = ?,
                        coverage_snaps = ?,
                        targets_allowed = ?,
                        receptions_allowed = ?,
                        yards_allowed = ?,
                        yac_allowed = ?,
                        ypr_allowed = ?,
                        tds_allowed = ?,
                        alignment_boundary_pct = ?,
                        alignment_slot_pct = ?,
                        alignment_deep_pct = ?,
                        alignment_box_pct = ?,
                        man_coverage_pct = ?,
                        zone_coverage_pct = ?,
                        penalties = ?,
                        pass_breakups = ?,
                        interceptions = ?,
                        raw_json = ?,
                        ingested_at = ?
                    WHERE defender_id = ? AND game_id = ?
                """, (
                    stats["snaps"], stats["coverage_snaps"],
                    stats["targets_allowed"], stats["receptions_allowed"],
                    stats["yards_allowed"], stats["yac_allowed"],
                    stats["ypr_allowed"], stats["tds_allowed"],
                    stats["alignment_boundary_pct"], stats["alignment_slot_pct"],
                    stats["alignment_deep_pct"], stats["alignment_box_pct"],
                    stats["man_coverage_pct"], stats["zone_coverage_pct"],
                    stats["penalties"], stats["pass_breakups"],
                    stats["interceptions"], stats["raw_json"],
                    timestamp, stats["defender_id"], stats["game_id"]
                ))
                self.stats["updated"] += 1
            else:
                cursor.execute("""
                    INSERT INTO defender_game_stats
                    (defender_id, game_id, season, week, snaps, coverage_snaps,
                     targets_allowed, receptions_allowed, yards_allowed,
                     yac_allowed, ypr_allowed, tds_allowed,
                     alignment_boundary_pct, alignment_slot_pct,
                     alignment_deep_pct, alignment_box_pct,
                     man_coverage_pct, zone_coverage_pct,
                     penalties, pass_breakups, interceptions,
                     raw_json, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    stats["defender_id"], stats["game_id"],
                    stats["season"], stats["week"],
                    stats["snaps"], stats["coverage_snaps"],
                    stats["targets_allowed"], stats["receptions_allowed"],
                    stats["yards_allowed"], stats["yac_allowed"],
                    stats["ypr_allowed"], stats["tds_allowed"],
                    stats["alignment_boundary_pct"], stats["alignment_slot_pct"],
                    stats["alignment_deep_pct"], stats["alignment_box_pct"],
                    stats["man_coverage_pct"], stats["zone_coverage_pct"],
                    stats["penalties"], stats["pass_breakups"],
                    stats["interceptions"], stats["raw_json"],
                    timestamp
                ))
                self.stats["inserted"] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error inserting defender stats: {e}")
            self.stats["errors"] += 1
            return False
    
    def create_placeholder_records_for_week(self, season: int, week: int) -> int:
        """
        Create placeholder defender stats records for defenders who played.
        
        This creates empty records that can be populated later when
        detailed data becomes available.
        
        Args:
            season: Season year
            week: Week number
        
        Returns:
            Number of records created
        """
        logger.info(f"Creating placeholder defender records for {season} week {week}...")
        
        # Get games for this week
        games = self.db.fetch_all(
            "SELECT game_id, home_team_id, away_team_id FROM games WHERE season = ? AND week = ?",
            (season, week)
        )
        
        if not games:
            return 0
        
        created = 0
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for game in games:
                game_id = game["game_id"]
                teams = [game["home_team_id"], game["away_team_id"]]
                
                # Get defenders for both teams
                for team_id in teams:
                    for defender_id, info in self._defenders.items():
                        if info["team_id"] == team_id:
                            # Check if record already exists
                            cursor.execute(
                                "SELECT defender_id FROM defender_game_stats WHERE defender_id = ? AND game_id = ?",
                                (defender_id, game_id)
                            )
                            if cursor.fetchone():
                                continue
                            
                            # Create placeholder
                            stats = self.create_placeholder_stats(
                                defender_id, game_id, season, week
                            )
                            
                            if self.insert_defender_stats(stats, cursor):
                                created += 1
        
        logger.info(f"Created {created} placeholder defender records")
        return created
    
    def validate_stats(self) -> Dict[str, Any]:
        """Validate defender stats."""
        logger.info("Validating defender stats...")
        
        results = {
            "total_records": self.db.get_row_count("defender_game_stats"),
            "by_season": {},
            "by_position_group": {},
            "validation_passed": True,
            "issues": []
        }
        
        # Count by season
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT season, COUNT(*) as count
                FROM defender_game_stats
                GROUP BY season
            """)
            for row in cursor.fetchall():
                results["by_season"][row["season"]] = row["count"]
        
        # Count by position group
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT d.position_group, COUNT(*) as count
                FROM defender_game_stats dgs
                JOIN defenders d ON dgs.defender_id = d.defender_id
                GROUP BY d.position_group
            """)
            for row in cursor.fetchall():
                results["by_position_group"][row["position_group"]] = row["count"]
        
        # Check for orphan records
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM defender_game_stats dgs
                WHERE dgs.defender_id NOT IN (SELECT defender_id FROM defenders)
            """)
            result = cursor.fetchone()
            if result and result["count"] > 0:
                results["issues"].append(f"{result['count']} records with missing defender")
                results["validation_passed"] = False
        
        # Validate coverage percentages
        with self.db.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM defender_game_stats
                WHERE man_coverage_pct > 0 AND zone_coverage_pct > 0
                AND ABS((man_coverage_pct + zone_coverage_pct) - 1.0) > 0.05
            """)
            result = cursor.fetchone()
            if result and result["count"] > 0:
                results["issues"].append(f"{result['count']} records with invalid coverage sum")
        
        return results
    
    def run(
        self,
        csv_file: Optional[Path] = None,
        create_placeholders: bool = False,
        seasons: Optional[List[int]] = None,
    ) -> Dict:
        """
        Run defender stats ingestion.

        Args:
            csv_file: Optional CSV file to load (bypasses Sleeper)
            create_placeholders: If True, create zero-filled placeholder records
            seasons: Seasons to process (None = rolling window)

        Returns:
            Ingestion results
        """
        logger.info("=" * 60)
        logger.info("STARTING DEFENDER STATS INGESTION")
        logger.info("=" * 60)

        start_time = time.time()

        # Determine which seasons to process
        if seasons:
            seasons_to_process: List[int] = seasons
        else:
            seasons_to_process = list(
                range(
                    CURRENT_SEASON - ROLLING_WINDOW_SEASONS + 1,
                    CURRENT_SEASON + 1,
                )
            )

        results_by_season: Dict[int, int] = {}

        if csv_file and csv_file.exists():
            # Legacy path: load from a prepared CSV file
            season = seasons_to_process[0]
            logger.info(
                f"Loading defender stats from CSV for season {season}: {csv_file}"
            )
            self.load_from_csv(csv_file, season)
            results_by_season[season] = (
                self.stats["inserted"] + self.stats["updated"]
            )
        else:
            # Default path: pull defender stats directly from Sleeper API
            for season in seasons_to_process:
                logger.info(f"Ingesting defender stats for season {season}...")
                total_processed = 0
                consecutive_empty = 0

                for week in range(1, MAX_REGULAR_SEASON_WEEKS + 1):
                    processed = self.ingest_week(season, week)
                    total_processed += processed

                    if processed == 0:
                        consecutive_empty += 1
                        if consecutive_empty >= 3:
                            logger.info(
                                f"End of {season} defender stats detected at "
                                f"week {week}"
                            )
                            break
                    else:
                        consecutive_empty = 0

                    # Light rate limiting to be polite to the API
                    time.sleep(0.5)

                results_by_season[season] = total_processed

        # Optionally backfill placeholder rows for defenders/games without stats
        if create_placeholders:
            logger.info("Creating placeholder defender records where missing...")
            for season in seasons_to_process:
                for week in range(1, MAX_REGULAR_SEASON_WEEKS + 1):
                    count = self.db.fetch_one(
                        "SELECT COUNT(*) as c FROM games WHERE season = ? AND week = ?",
                        (season, week),
                    )
                    if count and count["c"] > 0:
                        self.create_placeholder_records_for_week(season, week)

        duration = time.time() - start_time
        validation = self.validate_stats()

        result = {
            "inserted": self.stats["inserted"],
            "updated": self.stats["updated"],
            "skipped": self.stats["skipped"],
            "errors": self.stats["errors"],
            "by_season": results_by_season,
            "duration_seconds": round(duration, 2),
            "validation": validation,
        }

        logger.log_ingestion_complete(
            row_count=self.stats["inserted"] + self.stats["updated"],
            duration_seconds=duration,
        )

        logger.info("=" * 60)
        logger.info("DEFENDER STATS INGESTION COMPLETE")
        logger.info("=" * 60)

        return result

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest defender stats")
    parser.add_argument("--csv", type=Path, help="CSV file to load")
    parser.add_argument("--create-placeholders", action="store_true",
                       help="Create placeholder records")
    parser.add_argument("--season", type=int, nargs="+", help="Season(s) to process")
    parser.add_argument("--validate-only", action="store_true")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Phase 1")
    print("Defender Stats Ingestion (Step 6b)")
    print(f"{'='*60}\n")
    
    ingestion = DefenderStatsIngestion()
    
    if args.validate_only:
        validation = ingestion.validate_stats()
        print(f"\nValidation: {validation}")
        return 0 if validation['validation_passed'] else 1
    
    result = ingestion.run(
        csv_file=args.csv,
        create_placeholders=args.create_placeholders,
        seasons=args.season
    )
    
    print(f"\nResults: inserted={result['inserted']}, updated={result['updated']}")
    print(f"Duration: {result['duration_seconds']}s")
    print(f"Log file: {logger.log_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())
