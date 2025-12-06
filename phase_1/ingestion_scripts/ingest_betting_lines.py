"""
Betting Lines Ingestion for NFL Fantasy Prediction Engine

This module fetches betting lines from The Odds API and stores them
in the betting_lines table for analysis and bet tracking.

Supports:
- Featured markets (moneyline, spreads, totals)
- Player prop markets (passing/rushing/receiving yards, TDs, receptions)
- Current odds and historical snapshots
- Multiple bookmakers with FanDuel as primary

API Cost:
- Featured markets: 1 credit per market per region (~3 credits/call)
- Player props: 1 credit per market per region per event (~96 credits for 16 games)
- Historical: 10x normal cost

Author: NFL Fantasy Prediction Engine Team
Phase: 1 Extension - Betting Integration
Version: 2.0
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Import settings
try:
    from config.settings_odds_api import (
        ODDS_API_KEY,
        ODDS_API_BASE_URL,
        ODDS_API_SPORT,
        ODDS_API_REGIONS,
        ODDS_API_PREFERRED_BOOKMAKER,
        ODDS_API_BOOKMAKERS,
        ODDS_API_FEATURED_MARKETS,
        ODDS_API_PLAYER_PROP_MARKETS,
        ODDS_API_MARKET_TO_STAT_KEY,
        ODDS_API_TEAM_NAME_TO_ABBREV,
        ODDS_API_TIMEOUT,
        ODDS_API_RETRY_ATTEMPTS,
        ODDS_API_RETRY_DELAY,
        get_odds_api_endpoints,
        american_odds_to_decimal,
        estimate_nfl_week_from_date,
    )
except ImportError:
    # Minimal fallback if settings not available
    import os
    ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
    ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
    ODDS_API_SPORT = "americanfootball_nfl"
    ODDS_API_REGIONS = "us"
    ODDS_API_PREFERRED_BOOKMAKER = "fanduel"
    ODDS_API_TIMEOUT = 30
    ODDS_API_RETRY_ATTEMPTS = 3
    ODDS_API_RETRY_DELAY = 2

from config.settings import CURRENT_SEASON

# Initialize logger
logger = get_ingestion_logger("ingest_betting_lines")


class BettingLinesIngestion:
    """
    Ingests betting lines from The Odds API into the database.
    
    Handles:
    - Fetching current and historical odds
    - Parsing featured markets and player props
    - Matching games and players to existing database records
    - Deduplication and storage
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the ingestion class.
        
        Args:
            db: Optional database connection
        """
        self.db = db or get_db()
        self.endpoints = get_odds_api_endpoints() if 'get_odds_api_endpoints' in dir() else self._default_endpoints()
        
        # Stats tracking
        self.stats = {
            'api_calls': 0,
            'api_credits_used': 0,
            'lines_inserted': 0,
            'lines_skipped': 0,
            'events_processed': 0,
            'errors': [],
        }
        
        # Caches
        self._game_cache = {}
        self._player_cache = {}
    
    def _default_endpoints(self) -> Dict:
        """Default endpoint configuration."""
        return {
            "odds": f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/odds",
            "events": f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/events",
            "event_odds": lambda eid: f"{ODDS_API_BASE_URL}/sports/{ODDS_API_SPORT}/events/{eid}/odds",
            "historical_odds": f"{ODDS_API_BASE_URL}/historical/sports/{ODDS_API_SPORT}/odds",
        }
    
    def _make_request(
        self,
        url: str,
        params: Dict,
        retry_count: int = 0,
    ) -> Tuple[Optional[Dict], Dict]:
        """
        Make an API request with retry logic.
        
        Args:
            url: API endpoint URL
            params: Query parameters
            retry_count: Current retry attempt
            
        Returns:
            Tuple of (response_data, headers)
        """
        params['apiKey'] = ODDS_API_KEY
        
        try:
            response = requests.get(url, params=params, timeout=ODDS_API_TIMEOUT)
            self.stats['api_calls'] += 1
            
            # Track API usage from headers
            if 'x-requests-used' in response.headers:
                self.stats['api_credits_used'] = int(response.headers.get('x-requests-used', 0))
            
            if response.status_code == 200:
                return response.json(), dict(response.headers)
            elif response.status_code == 429:
                # Rate limited
                logger.warning("Rate limited by Odds API")
                if retry_count < ODDS_API_RETRY_ATTEMPTS:
                    time.sleep(ODDS_API_RETRY_DELAY * (retry_count + 1))
                    return self._make_request(url, params, retry_count + 1)
            elif response.status_code == 401:
                logger.error("Invalid API key")
                self.stats['errors'].append("Invalid API key")
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                self.stats['errors'].append(f"API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.warning(f"Request timeout, retry {retry_count + 1}")
            if retry_count < ODDS_API_RETRY_ATTEMPTS:
                time.sleep(ODDS_API_RETRY_DELAY)
                return self._make_request(url, params, retry_count + 1)
            self.stats['errors'].append("Request timeout")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            self.stats['errors'].append(str(e))
        
        return None, {}
    
    def _determine_season_week(self, commence_time: str) -> Tuple[int, int]:
        """
        Determine NFL season and week from game commence time.
        
        Args:
            commence_time: ISO timestamp of game start
            
        Returns:
            Tuple of (season, week)
        """
        try:
            dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            
            # Use estimation function if available
            if 'estimate_nfl_week_from_date' in dir():
                return estimate_nfl_week_from_date(dt)
            
            # Basic fallback
            season = dt.year if dt.month >= 9 else dt.year - 1
            
            # Rough week calculation
            if dt.month >= 9:
                weeks_since_sept = ((dt - datetime(dt.year, 9, 1, tzinfo=dt.tzinfo)).days) // 7
                week = min(weeks_since_sept + 1, 18)
            elif dt.month <= 2:
                week = 18  # Playoffs/Super Bowl
            else:
                week = 0
            
            return season, week
            
        except Exception as e:
            logger.warning(f"Could not parse commence_time: {commence_time}, error: {e}")
            return CURRENT_SEASON, 0
    
    def _match_game_id(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
    ) -> Optional[str]:
        """
        Match Odds API event to our games table.
        
        Args:
            home_team: Home team name from Odds API
            away_team: Away team name from Odds API
            season: NFL season
            week: NFL week
            
        Returns:
            game_id if found, None otherwise
        """
        cache_key = (home_team, away_team, season, week)
        if cache_key in self._game_cache:
            return self._game_cache[cache_key]
        
        # Convert team names to abbreviations
        home_abbrev = ODDS_API_TEAM_NAME_TO_ABBREV.get(home_team)
        away_abbrev = ODDS_API_TEAM_NAME_TO_ABBREV.get(away_team)
        
        if not home_abbrev or not away_abbrev:
            logger.warning(f"Could not map team names: {home_team} vs {away_team}")
            return None
        
        # Query games table
        result = self.db.fetch_one("""
            SELECT game_id FROM games
            WHERE season = ? AND week = ? AND home_team_id = ? AND away_team_id = ?
        """, (season, week, home_abbrev, away_abbrev))
        
        game_id = result['game_id'] if result else None
        self._game_cache[cache_key] = game_id
        
        return game_id
    
    def _match_player_id(self, player_name: str) -> Optional[str]:
        """
        Match player name from Odds API to our players table.
        
        Args:
            player_name: Player name from Odds API
            
        Returns:
            player_id if found, None otherwise
        """
        if player_name in self._player_cache:
            return self._player_cache[player_name]
        
        # Try exact match first
        result = self.db.fetch_one("""
            SELECT player_id FROM players WHERE full_name = ?
        """, (player_name,))
        
        if result:
            self._player_cache[player_name] = result['player_id']
            return result['player_id']
        
        # Try fuzzy match on last name
        parts = player_name.split()
        if len(parts) >= 2:
            last_name = parts[-1]
            first_initial = parts[0][0] if parts[0] else ''
            
            result = self.db.fetch_one("""
                SELECT player_id, full_name FROM players 
                WHERE full_name LIKE ? AND full_name LIKE ?
                LIMIT 1
            """, (f"{first_initial}%", f"% {last_name}"))
            
            if result:
                self._player_cache[player_name] = result['player_id']
                return result['player_id']
        
        self._player_cache[player_name] = None
        return None
    
    def _insert_betting_line(
        self,
        cursor,
        line_data: Dict,
        is_historical: bool = False,
    ) -> bool:
        """
        Insert a betting line record.
        
        Args:
            cursor: Database cursor
            line_data: Dictionary with line data
            is_historical: Whether this is historical data
            
        Returns:
            True if inserted, False if skipped
        """
        try:
            cursor.execute("""
                INSERT INTO betting_lines (
                    odds_api_event_id, game_id, season, week,
                    home_team, away_team, commence_time,
                    market_type, bookmaker, bookmaker_title,
                    player_name, player_id,
                    outcome_name, outcome_description, line_value,
                    odds_american, odds_decimal,
                    odds_api_last_update, snapshot_timestamp, is_historical
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                line_data['odds_api_event_id'],
                line_data.get('game_id'),
                line_data['season'],
                line_data['week'],
                line_data['home_team'],
                line_data['away_team'],
                line_data.get('commence_time'),
                line_data['market_type'],
                line_data['bookmaker'],
                line_data.get('bookmaker_title'),
                line_data.get('player_name'),
                line_data.get('player_id'),
                line_data['outcome_name'],
                line_data.get('outcome_description'),
                line_data.get('line_value'),
                line_data.get('odds_american'),
                line_data.get('odds_decimal'),
                line_data.get('odds_api_last_update'),
                line_data.get('snapshot_timestamp', datetime.now().isoformat()),
                1 if is_historical else 0,
            ))
            self.stats['lines_inserted'] += 1
            return True
            
        except Exception as e:
            logger.debug(f"Could not insert line: {e}")
            self.stats['lines_skipped'] += 1
            return False
    
    def _parse_featured_odds(
        self,
        event: Dict,
        bookmakers: List[Dict],
        snapshot_time: Optional[str] = None,
    ) -> List[Dict]:
        """
        Parse featured market odds (h2h, spreads, totals).
        
        Args:
            event: Event data from API
            bookmakers: List of bookmaker data
            snapshot_time: Optional snapshot timestamp
            
        Returns:
            List of parsed line dictionaries
        """
        lines = []
        
        home_team = event.get('home_team', '')
        away_team = event.get('away_team', '')
        commence_time = event.get('commence_time', '')
        event_id = event.get('id', '')
        
        season, week = self._determine_season_week(commence_time)
        game_id = self._match_game_id(home_team, away_team, season, week)
        
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get('key', '')
            bookmaker_title = bookmaker.get('title', '')
            last_update = bookmaker.get('last_update', '')
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                
                for outcome in market.get('outcomes', []):
                    line_value = outcome.get('point')
                    odds_american = outcome.get('price')
                    odds_decimal = american_odds_to_decimal(odds_american) if odds_american else None
                    
                    lines.append({
                        'odds_api_event_id': event_id,
                        'game_id': game_id,
                        'season': season,
                        'week': week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': commence_time,
                        'market_type': market_key,
                        'bookmaker': bookmaker_key,
                        'bookmaker_title': bookmaker_title,
                        'outcome_name': outcome.get('name', ''),
                        'outcome_description': outcome.get('description'),
                        'line_value': line_value,
                        'odds_american': odds_american,
                        'odds_decimal': odds_decimal,
                        'odds_api_last_update': last_update,
                        'snapshot_timestamp': snapshot_time or datetime.now().isoformat(),
                    })
        
        return lines
    
    def _parse_player_props(
        self,
        event: Dict,
        bookmakers: List[Dict],
        snapshot_time: Optional[str] = None,
    ) -> List[Dict]:
        """
        Parse player prop market odds.
        
        Args:
            event: Event data from API
            bookmakers: List of bookmaker data
            snapshot_time: Optional snapshot timestamp
            
        Returns:
            List of parsed line dictionaries
        """
        lines = []
        
        home_team = event.get('home_team', '')
        away_team = event.get('away_team', '')
        commence_time = event.get('commence_time', '')
        event_id = event.get('id', '')
        
        season, week = self._determine_season_week(commence_time)
        game_id = self._match_game_id(home_team, away_team, season, week)
        
        for bookmaker in bookmakers:
            bookmaker_key = bookmaker.get('key', '')
            bookmaker_title = bookmaker.get('title', '')
            last_update = bookmaker.get('last_update', '')
            
            for market in bookmaker.get('markets', []):
                market_key = market.get('key', '')
                
                for outcome in market.get('outcomes', []):
                    player_name = outcome.get('description') or outcome.get('name', '')
                    player_id = self._match_player_id(player_name)
                    
                    line_value = outcome.get('point')
                    odds_american = outcome.get('price')
                    odds_decimal = american_odds_to_decimal(odds_american) if odds_american else None
                    
                    lines.append({
                        'odds_api_event_id': event_id,
                        'game_id': game_id,
                        'season': season,
                        'week': week,
                        'home_team': home_team,
                        'away_team': away_team,
                        'commence_time': commence_time,
                        'market_type': market_key,
                        'bookmaker': bookmaker_key,
                        'bookmaker_title': bookmaker_title,
                        'player_name': player_name,
                        'player_id': player_id,
                        'outcome_name': outcome.get('name', ''),  # Over/Under
                        'outcome_description': player_name,
                        'line_value': line_value,
                        'odds_american': odds_american,
                        'odds_decimal': odds_decimal,
                        'odds_api_last_update': last_update,
                        'snapshot_timestamp': snapshot_time or datetime.now().isoformat(),
                    })
        
        return lines
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def fetch_current_featured_odds(
        self,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fetch current featured market odds (moneyline, spreads, totals).
        
        Args:
            markets: List of markets to fetch (default: h2h, spreads, totals)
            bookmakers: List of bookmakers (default: all US)
            
        Returns:
            Dictionary with fetch results
        """
        if not ODDS_API_KEY:
            logger.error("ODDS_API_KEY not configured")
            return {'success': False, 'error': 'API key not configured'}
        
        markets = markets or ['h2h', 'spreads', 'totals']
        
        params = {
            'regions': ODDS_API_REGIONS,
            'markets': ','.join(markets),
            'oddsFormat': 'american',
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        logger.info(f"Fetching featured odds for markets: {markets}")
        
        data, headers = self._make_request(self.endpoints['odds'], params)
        
        if not data:
            return {'success': False, 'error': 'No data returned'}
        
        # Process events
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for event in data:
                lines = self._parse_featured_odds(event, event.get('bookmakers', []))
                for line in lines:
                    self._insert_betting_line(cursor, line)
                self.stats['events_processed'] += 1
        
        logger.info(
            f"Fetched featured odds: {self.stats['events_processed']} events, "
            f"{self.stats['lines_inserted']} lines",
            event="featured_odds_complete"
        )
        
        return {
            'success': True,
            'events_processed': self.stats['events_processed'],
            'lines_inserted': self.stats['lines_inserted'],
            'api_credits_used': self.stats['api_credits_used'],
        }
    
    def fetch_current_player_props(
        self,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
    ) -> Dict:
        """
        Fetch current player prop odds.
        
        Args:
            markets: List of prop markets to fetch
            bookmakers: List of bookmakers
            event_ids: Specific event IDs (if None, fetches for all events)
            
        Returns:
            Dictionary with fetch results
        """
        if not ODDS_API_KEY:
            logger.error("ODDS_API_KEY not configured")
            return {'success': False, 'error': 'API key not configured'}
        
        markets = markets or [
            'player_pass_yds', 'player_rush_yds', 'player_reception_yds',
            'player_receptions', 'player_pass_tds', 'player_anytime_td'
        ]
        
        # Get event list first if not provided
        if not event_ids:
            events_data, _ = self._make_request(
                self.endpoints['events'],
                {'regions': ODDS_API_REGIONS}
            )
            if events_data:
                event_ids = [e['id'] for e in events_data]
            else:
                return {'success': False, 'error': 'Could not fetch events'}
        
        logger.info(f"Fetching player props for {len(event_ids)} events")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for event_id in event_ids:
                endpoint = self.endpoints['event_odds'](event_id)
                
                params = {
                    'regions': ODDS_API_REGIONS,
                    'markets': ','.join(markets),
                    'oddsFormat': 'american',
                }
                
                if bookmakers:
                    params['bookmakers'] = ','.join(bookmakers)
                
                data, _ = self._make_request(endpoint, params)
                
                if data:
                    lines = self._parse_player_props(data, data.get('bookmakers', []))
                    for line in lines:
                        self._insert_betting_line(cursor, line)
                    self.stats['events_processed'] += 1
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
        
        logger.info(
            f"Fetched player props: {self.stats['events_processed']} events, "
            f"{self.stats['lines_inserted']} lines",
            event="player_props_complete"
        )
        
        return {
            'success': True,
            'events_processed': self.stats['events_processed'],
            'lines_inserted': self.stats['lines_inserted'],
            'api_credits_used': self.stats['api_credits_used'],
        }
    
    def run_weekly_pull(
        self,
        include_props: bool = True,
        bookmakers: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run a complete weekly odds pull.
        
        This is the recommended method for regular weekly updates.
        Best run on Wednesday evening when lines are stable.
        
        Args:
            include_props: Whether to include player props
            bookmakers: Specific bookmakers to use
            
        Returns:
            Dictionary with combined results
        """
        logger.info("Starting weekly odds pull")
        
        results = {
            'featured': None,
            'props': None,
            'total_lines': 0,
            'total_credits': 0,
        }
        
        # Fetch featured markets
        results['featured'] = self.fetch_current_featured_odds(bookmakers=bookmakers)
        results['total_lines'] += results['featured'].get('lines_inserted', 0)
        
        # Fetch player props if requested
        if include_props:
            results['props'] = self.fetch_current_player_props(bookmakers=bookmakers)
            results['total_lines'] += results['props'].get('lines_inserted', 0)
        
        results['total_credits'] = self.stats['api_credits_used']
        
        logger.info(
            f"Weekly pull complete: {results['total_lines']} lines, "
            f"{results['total_credits']} credits used",
            event="weekly_pull_complete"
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get current ingestion statistics."""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset ingestion statistics."""
        self.stats = {
            'api_calls': 0,
            'api_credits_used': 0,
            'lines_inserted': 0,
            'lines_skipped': 0,
            'events_processed': 0,
            'errors': [],
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch NFL betting lines from The Odds API"
    )
    parser.add_argument(
        '--weekly',
        action='store_true',
        help='Run full weekly pull (featured + props)'
    )
    parser.add_argument(
        '--current-featured',
        action='store_true',
        help='Fetch current featured markets only'
    )
    parser.add_argument(
        '--current-props',
        action='store_true',
        help='Fetch current player props only'
    )
    parser.add_argument(
        '--bookmaker',
        type=str,
        default=None,
        help='Specific bookmaker to use (default: all)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("NFL Fantasy Prediction Engine - Betting Lines Ingestion")
    print(f"{'='*60}")
    
    if not ODDS_API_KEY:
        print("\nERROR: ODDS_API_KEY environment variable not set!")
        print("Set it with: export ODDS_API_KEY='your_key_here'")
        return 1
    
    db = get_db()
    ingestion = BettingLinesIngestion(db)
    
    bookmakers = [args.bookmaker] if args.bookmaker else None
    
    if args.weekly:
        result = ingestion.run_weekly_pull(bookmakers=bookmakers)
        print(f"\nWeekly Pull Results:")
        print(f"  Total lines: {result['total_lines']}")
        print(f"  API credits used: {result['total_credits']}")
        
    elif args.current_featured:
        result = ingestion.fetch_current_featured_odds(bookmakers=bookmakers)
        print(f"\nFeatured Odds Results:")
        print(f"  Events processed: {result.get('events_processed', 0)}")
        print(f"  Lines inserted: {result.get('lines_inserted', 0)}")
        
    elif args.current_props:
        result = ingestion.fetch_current_player_props(bookmakers=bookmakers)
        print(f"\nPlayer Props Results:")
        print(f"  Events processed: {result.get('events_processed', 0)}")
        print(f"  Lines inserted: {result.get('lines_inserted', 0)}")
        
    else:
        parser.print_help()
        return 0
    
    stats = ingestion.get_stats()
    if stats['errors']:
        print(f"\nErrors: {stats['errors']}")
    
    print(f"\nLog file: {logger.log_file}")
    return 0


if __name__ == "__main__":
    exit(main())
