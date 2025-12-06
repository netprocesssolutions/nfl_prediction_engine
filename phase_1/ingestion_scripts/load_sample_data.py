"""
Sample Data Loader for NFL Fantasy Prediction Engine - Phase 1

This module provides sample data for testing when the Sleeper API is not accessible.
It generates realistic sample data that mimics the structure of real NFL data.

This is for TESTING ONLY - real deployment should use the actual Sleeper API.

Author: NFL Fantasy Prediction Engine Team
Phase: 1 - Data Ingestion & Database Setup
Version: 2.0
"""

import sys
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import NFL_TEAMS, OFFENSIVE_POSITIONS, DEFENSIVE_POSITIONS
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

logger = get_ingestion_logger("sample_data")

# Sample player names by position
SAMPLE_PLAYERS = {
    "QB": [
        ("Patrick Mahomes", "KC"), ("Josh Allen", "BUF"), ("Jalen Hurts", "PHI"),
        ("Lamar Jackson", "BAL"), ("Joe Burrow", "CIN"), ("Justin Herbert", "LAC"),
        ("Dak Prescott", "DAL"), ("Jared Goff", "DET"), ("Tua Tagovailoa", "MIA"),
        ("C.J. Stroud", "HOU"), ("Jordan Love", "GB"), ("Brock Purdy", "SF"),
        ("Kyler Murray", "ARI"), ("Trevor Lawrence", "JAX"), ("Geno Smith", "SEA"),
        ("Kirk Cousins", "ATL"), ("Derek Carr", "NO"), ("Matthew Stafford", "LAR"),
    ],
    "RB": [
        ("Christian McCaffrey", "SF"), ("Breece Hall", "NYJ"), ("Bijan Robinson", "ATL"),
        ("Jahmyr Gibbs", "DET"), ("Derrick Henry", "BAL"), ("Saquon Barkley", "PHI"),
        ("Jonathan Taylor", "IND"), ("Travis Etienne", "JAX"), ("Josh Jacobs", "GB"),
        ("De'Von Achane", "MIA"), ("Kyren Williams", "LAR"), ("Alvin Kamara", "NO"),
        ("Rachaad White", "TB"), ("James Cook", "BUF"), ("Isiah Pacheco", "KC"),
        ("Tony Pollard", "TEN"), ("Aaron Jones", "MIN"), ("Nick Chubb", "CLE"),
    ],
    "WR": [
        ("Tyreek Hill", "MIA"), ("CeeDee Lamb", "DAL"), ("Ja'Marr Chase", "CIN"),
        ("Amon-Ra St. Brown", "DET"), ("A.J. Brown", "PHI"), ("Davante Adams", "LV"),
        ("Garrett Wilson", "NYJ"), ("Chris Olave", "NO"), ("DK Metcalf", "SEA"),
        ("Jaylen Waddle", "MIA"), ("Brandon Aiyuk", "SF"), ("DeVonta Smith", "PHI"),
        ("Mike Evans", "TB"), ("Stefon Diggs", "HOU"), ("Deebo Samuel", "SF"),
        ("Amari Cooper", "CLE"), ("Michael Pittman", "IND"), ("Drake London", "ATL"),
        ("Terry McLaurin", "WAS"), ("DJ Moore", "CHI"), ("Puka Nacua", "LAR"),
        ("Nico Collins", "HOU"), ("Rashee Rice", "KC"), ("Keenan Allen", "CHI"),
    ],
    "TE": [
        ("Travis Kelce", "KC"), ("Sam LaPorta", "DET"), ("T.J. Hockenson", "MIN"),
        ("George Kittle", "SF"), ("Mark Andrews", "BAL"), ("Dallas Goedert", "PHI"),
        ("Evan Engram", "JAX"), ("David Njoku", "CLE"), ("Kyle Pitts", "ATL"),
        ("Jake Ferguson", "DAL"), ("Dalton Kincaid", "BUF"), ("Pat Freiermuth", "PIT"),
    ],
}

SAMPLE_DEFENDERS = {
    "CB": [
        ("Sauce Gardner", "NYJ"), ("Patrick Surtain II", "DEN"), ("Devon Witherspoon", "SEA"),
        ("Trent McDuffie", "KC"), ("Jaire Alexander", "GB"), ("Jaylon Johnson", "CHI"),
        ("Denzel Ward", "CLE"), ("Jalen Ramsey", "MIA"), ("Derek Stingley Jr.", "HOU"),
        ("Christian Gonzalez", "NE"), ("Marshon Lattimore", "NO"), ("Donte Jackson", "PIT"),
    ],
    "S": [
        ("Kyle Hamilton", "BAL"), ("Jevon Holland", "MIA"), ("Antoine Winfield Jr.", "TB"),
        ("Derwin James", "LAC"), ("Jessie Bates III", "ATL"), ("Minkah Fitzpatrick", "PIT"),
        ("Jordan Poyer", "BUF"), ("Harrison Smith", "MIN"), ("Xavier McKinney", "GB"),
        ("Justin Simmons", "DEN"), ("Budda Baker", "ARI"), ("Marcus Williams", "BAL"),
    ],
    "LB": [
        ("Micah Parsons", "DAL"), ("T.J. Watt", "PIT"), ("Fred Warner", "SF"),
        ("Roquan Smith", "BAL"), ("Matt Milano", "BUF"), ("Foyesade Oluokun", "JAX"),
        ("Lavonte David", "TB"), ("Demario Davis", "NO"), ("Tremaine Edmunds", "CHI"),
        ("Bobby Wagner", "WAS"), ("Shaquille Leonard", "IND"), ("Zaire Franklin", "IND"),
    ],
}


def generate_player_id(name: str) -> str:
    """Generate a consistent player ID from name."""
    return f"p_{abs(hash(name)) % 10000000}"


def generate_sample_players(db: DatabaseConnection) -> int:
    """Generate and insert sample players."""
    logger.info("Generating sample players...")
    
    inserted = 0
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        for position, players in SAMPLE_PLAYERS.items():
            for name, team in players:
                player_id = generate_player_id(name)
                
                # Check if exists
                cursor.execute("SELECT player_id FROM players WHERE player_id = ?", (player_id,))
                if cursor.fetchone():
                    continue
                
                # Generate random physical attributes
                if position == "QB":
                    height = random.randint(73, 78)
                    weight = random.randint(210, 240)
                elif position == "RB":
                    height = random.randint(68, 73)
                    weight = random.randint(195, 230)
                elif position == "WR":
                    height = random.randint(69, 76)
                    weight = random.randint(175, 220)
                else:  # TE
                    height = random.randint(74, 78)
                    weight = random.randint(240, 265)
                
                cursor.execute("""
                    INSERT INTO players (player_id, full_name, position, team_id, 
                                        height, weight, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """, (player_id, name, position, team, height, weight, timestamp, timestamp))
                inserted += 1
    
    logger.info(f"Generated {inserted} sample players")
    return inserted


def generate_sample_defenders(db: DatabaseConnection) -> int:
    """Generate and insert sample defenders."""
    logger.info("Generating sample defenders...")
    
    inserted = 0
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        for position_group, defenders in SAMPLE_DEFENDERS.items():
            for name, team in defenders:
                defender_id = generate_player_id(f"def_{name}")
                
                cursor.execute("SELECT defender_id FROM defenders WHERE defender_id = ?", (defender_id,))
                if cursor.fetchone():
                    continue
                
                # Generate physical attributes
                if position_group == "CB":
                    height = random.randint(69, 74)
                    weight = random.randint(180, 200)
                    role = random.choice(["boundary", "slot"])
                elif position_group == "S":
                    height = random.randint(70, 74)
                    weight = random.randint(195, 215)
                    role = random.choice(["deep", "box"])
                else:  # LB
                    height = random.randint(72, 76)
                    weight = random.randint(230, 255)
                    role = random.choice(["box", "boundary"])
                
                cursor.execute("""
                    INSERT INTO defenders (defender_id, full_name, team_id, position_group,
                                          role, height, weight, coverage_role, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (defender_id, name, team, position_group, role, height, weight,
                      random.choice(["man", "zone", "hybrid"]), timestamp, timestamp))
                inserted += 1
    
    logger.info(f"Generated {inserted} sample defenders")
    return inserted


def generate_sample_games(db: DatabaseConnection, season: int, weeks: int = 18) -> int:
    """Generate sample games for a season."""
    logger.info(f"Generating sample games for {season}...")
    
    teams = list(NFL_TEAMS.keys())
    inserted = 0
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        for week in range(1, weeks + 1):
            # Shuffle teams and pair them up
            random.shuffle(teams)
            
            for i in range(0, len(teams) - 1, 2):
                home_team = teams[i]
                away_team = teams[i + 1]
                
                # Sort for consistent game_id
                t1, t2 = sorted([home_team, away_team])
                game_id = f"{season}_{week:02d}_{t1}_{t2}"
                
                cursor.execute("SELECT game_id FROM games WHERE game_id = ?", (game_id,))
                if cursor.fetchone():
                    continue
                
                cursor.execute("""
                    INSERT INTO games (game_id, season, week, home_team_id, away_team_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (game_id, season, week, home_team, away_team, timestamp))
                inserted += 1
    
    logger.info(f"Generated {inserted} sample games")
    return inserted


def generate_sample_stats(db: DatabaseConnection, season: int, weeks: int = 4) -> int:
    """Generate sample player stats."""
    logger.info(f"Generating sample stats for {season} weeks 1-{weeks}...")
    
    inserted = 0
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Get players
    players = db.fetch_all("SELECT player_id, position, team_id FROM players")
    
    # Get games
    games = db.fetch_all(
        "SELECT game_id, season, week, home_team_id, away_team_id FROM games WHERE season = ? AND week <= ?",
        (season, weeks)
    )
    
    with db.get_connection() as conn:
        cursor = conn.cursor()
        
        for game in games:
            game_id = game["game_id"]
            week = game["week"]
            home_team = game["home_team_id"]
            away_team = game["away_team_id"]
            
            # Generate stats for players on both teams
            for player in players:
                player_id = player["player_id"]
                position = player["position"]
                team_id = player["team_id"]
                
                if team_id not in [home_team, away_team]:
                    continue
                
                opponent = away_team if team_id == home_team else home_team
                
                # Check if exists
                cursor.execute(
                    "SELECT player_id FROM player_game_stats WHERE player_id = ? AND game_id = ?",
                    (player_id, game_id)
                )
                if cursor.fetchone():
                    continue
                
                # Generate realistic stats based on position
                if position == "QB":
                    snaps = random.randint(50, 75)
                    pass_att = random.randint(25, 45)
                    completions = int(pass_att * random.uniform(0.55, 0.75))
                    pass_yards = random.randint(180, 350)
                    pass_tds = random.randint(0, 4)
                    ints = random.randint(0, 2)
                    carries = random.randint(2, 8)
                    rush_yards = random.randint(-5, 40)
                    stats = {
                        "snaps": snaps, "routes": 0, "carries": carries,
                        "rush_yards": rush_yards, "rush_tds": random.randint(0, 1),
                        "targets": 0, "receptions": 0, "rec_yards": 0, "rec_tds": 0,
                        "completions": completions, "pass_attempts": pass_att,
                        "pass_yards": pass_yards, "pass_tds": pass_tds,
                        "interceptions": ints, "fumbles": random.randint(0, 1)
                    }
                elif position == "RB":
                    snaps = random.randint(20, 55)
                    carries = random.randint(8, 25)
                    rush_yards = random.randint(20, 120)
                    targets = random.randint(1, 8)
                    receptions = int(targets * random.uniform(0.6, 0.9))
                    rec_yards = random.randint(5, 60)
                    stats = {
                        "snaps": snaps, "routes": random.randint(5, 20),
                        "carries": carries, "rush_yards": rush_yards,
                        "rush_tds": random.randint(0, 2),
                        "targets": targets, "receptions": receptions,
                        "rec_yards": rec_yards, "rec_tds": random.randint(0, 1),
                        "completions": 0, "pass_attempts": 0, "pass_yards": 0,
                        "pass_tds": 0, "interceptions": 0,
                        "fumbles": random.randint(0, 1)
                    }
                elif position == "WR":
                    snaps = random.randint(30, 70)
                    targets = random.randint(2, 14)
                    receptions = int(targets * random.uniform(0.5, 0.8))
                    rec_yards = random.randint(10, 150)
                    stats = {
                        "snaps": snaps, "routes": random.randint(20, 40),
                        "carries": random.randint(0, 1), "rush_yards": random.randint(0, 15),
                        "rush_tds": 0, "targets": targets, "receptions": receptions,
                        "rec_yards": rec_yards, "rec_tds": random.randint(0, 2),
                        "completions": 0, "pass_attempts": 0, "pass_yards": 0,
                        "pass_tds": 0, "interceptions": 0, "fumbles": 0
                    }
                else:  # TE
                    snaps = random.randint(25, 60)
                    targets = random.randint(1, 10)
                    receptions = int(targets * random.uniform(0.55, 0.85))
                    rec_yards = random.randint(5, 90)
                    stats = {
                        "snaps": snaps, "routes": random.randint(15, 35),
                        "carries": 0, "rush_yards": 0, "rush_tds": 0,
                        "targets": targets, "receptions": receptions,
                        "rec_yards": rec_yards, "rec_tds": random.randint(0, 1),
                        "completions": 0, "pass_attempts": 0, "pass_yards": 0,
                        "pass_tds": 0, "interceptions": 0, "fumbles": 0
                    }
                
                # Calculate fantasy points (half PPR)
                fpts = (
                    stats["pass_yards"] * 0.04 +
                    stats["pass_tds"] * 4 +
                    stats["interceptions"] * -2 +
                    stats["rush_yards"] * 0.1 +
                    stats["rush_tds"] * 6 +
                    stats["rec_yards"] * 0.1 +
                    stats["rec_tds"] * 6 +
                    stats["receptions"] * 0.5 +
                    stats["fumbles"] * -2
                )
                
                cursor.execute("""
                    INSERT INTO player_game_stats 
                    (player_id, game_id, team_id, opponent_team_id, season, week,
                     snaps, routes, carries, rush_yards, rush_tds,
                     targets, receptions, rec_yards, rec_tds,
                     completions, pass_attempts, pass_yards, pass_tds, interceptions,
                     fumbles, fantasy_points_sleeper, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_id, game_id, team_id, opponent, season, week,
                    stats["snaps"], stats["routes"], stats["carries"],
                    stats["rush_yards"], stats["rush_tds"],
                    stats["targets"], stats["receptions"], stats["rec_yards"], stats["rec_tds"],
                    stats["completions"], stats["pass_attempts"], stats["pass_yards"],
                    stats["pass_tds"], stats["interceptions"], stats["fumbles"],
                    round(fpts, 1), timestamp
                ))
                inserted += 1
    
    logger.info(f"Generated {inserted} sample stat records")
    return inserted


def load_sample_data(db: Optional[DatabaseConnection] = None,
                     season: int = 2024, weeks: int = 4) -> Dict[str, int]:
    """
    Load a complete set of sample data for testing.
    
    Args:
        db: Database connection
        season: Season year
        weeks: Number of weeks to generate
    
    Returns:
        Dictionary of counts by data type
    """
    db = db or get_db()
    
    logger.info("=" * 60)
    logger.info("LOADING SAMPLE DATA")
    logger.info("=" * 60)
    
    results = {
        "players": generate_sample_players(db),
        "defenders": generate_sample_defenders(db),
        "games": generate_sample_games(db, season, weeks),
        "stats": generate_sample_stats(db, season, weeks),
    }
    
    logger.info("=" * 60)
    logger.info(f"SAMPLE DATA LOADED: {results}")
    logger.info("=" * 60)
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load sample data for testing")
    parser.add_argument("--season", type=int, default=2024, help="Season year")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Sample Data Loader")
    print(f"{'='*60}\n")
    
    results = load_sample_data(season=args.season, weeks=args.weeks)
    
    print(f"\nResults:")
    for key, count in results.items():
        print(f"  {key}: {count}")
    
    print(f"\nLog file: {logger.log_file}")


if __name__ == "__main__":
    main()
