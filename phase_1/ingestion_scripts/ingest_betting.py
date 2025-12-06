"""
Betting Lines Ingestion from The Odds API

Fetches and stores betting lines for:
- Moneyline, spreads, totals (featured markets)
- Player props (passing/rushing/receiving yards, TDs, receptions)

Your plan: 20,000 requests/month
- Featured markets: ~3 credits per pull
- Player props: ~6 credits per event
- Weekly full pull: ~100-150 credits

Author: NFL Fantasy Prediction Engine Team
Version: 2.0
"""

import sys
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import (
    ODDS_API_KEY,
    ODDS_API_BASE_URL,
    ODDS_API_SPORT,
    ODDS_API_REGIONS,
    ODDS_API_TIMEOUT,
    ODDS_API_RETRY_ATTEMPTS,
    ODDS_API_RETRY_DELAY,
    ODDS_API_PREFERRED_BOOKMAKER,
    ODDS_API_FEATURED_MARKETS,
    ODDS_API_PLAYER_PROP_MARKETS,
    ODDS_API_TEAM_NAME_TO_ABBREV,
    CURRENT_SEASON,
    get_odds_api_endpoints,
    american_odds_to_decimal,
    estimate_nfl_week_from_date,
)

logger = get_ingestion_logger("ingest_betting")


class BettingLinesIngestion:
    """Ingest betting lines from The Odds API."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_db()
        self.endpoints = get_odds_api_endpoints()
        
        self.stats = {
            'api_calls': 0,
            'credits_used': 0,
            'credits_remaining': None,
            'lines_inserted': 0,
            'events_processed': 0,
            'errors': [],
        }
        
        self._game_cache = {}
        self._player_cache = {}
    
    def _make_request(self, url: str, params: Dict, retry_count: int = 0) -> Tuple[Optional[Dict], Dict]:
        """Make API request with retry logic."""
        if not ODDS_API_KEY:
            logger.error("ODDS_API_KEY not set!")
            return None, {}
        
        params['apiKey'] = ODDS_API_KEY
        
        try:
            response = requests.get(url, params=params, timeout=ODDS_API_TIMEOUT)
            self.stats['api_calls'] += 1
            
            # Track usage from headers
            if 'x-requests-used' in response.headers:
                self.stats['credits_used'] = int(response.headers.get('x-requests-used', 0))
            if 'x-requests-remaining' in response.headers:
                self.stats['credits_remaining'] = int(response.headers.get('x-requests-remaining', 0))
            
            if response.status_code == 200:
                return response.json(), dict(response.headers)
            elif response.status_code == 429:
                logger.warning("Rate limited")
                if retry_count < ODDS_API_RETRY_ATTEMPTS:
                    time.sleep(ODDS_API_RETRY_DELAY * (retry_count + 1))
                    return self._make_request(url, params, retry_count + 1)
            elif response.status_code == 401:
                logger.error("Invalid API key!")
                self.stats['errors'].append("Invalid API key")
            else:
                logger.error(f"API error: {response.status_code}")
                self.stats['errors'].append(f"HTTP {response.status_code}")
                
        except requests.exceptions.Timeout:
            if retry_count < ODDS_API_RETRY_ATTEMPTS:
                time.sleep(ODDS_API_RETRY_DELAY)
                return self._make_request(url, params, retry_count + 1)
            self.stats['errors'].append("Timeout")
        except Exception as e:
            logger.error(f"Request error: {e}")
            self.stats['errors'].append(str(e))
        
        return None, {}
    
    def _get_season_week(self, commence_time: str) -> Tuple[int, int]:
        """Get season and week from commence time."""
        try:
            dt = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
            return estimate_nfl_week_from_date(dt)
        except:
            return CURRENT_SEASON, 0
    
    def _match_game(self, home_team: str, away_team: str, season: int, week: int) -> Optional[str]:
        """Match to our games table."""
        cache_key = (home_team, away_team, season, week)
        if cache_key in self._game_cache:
            return self._game_cache[cache_key]
        
        home_abbrev = ODDS_API_TEAM_NAME_TO_ABBREV.get(home_team)
        away_abbrev = ODDS_API_TEAM_NAME_TO_ABBREV.get(away_team)
        
        if not home_abbrev or not away_abbrev:
            return None
        
        result = self.db.fetch_one("""
            SELECT game_id FROM games
            WHERE season = ? AND week = ? AND home_team_id = ? AND away_team_id = ?
        """, (season, week, home_abbrev, away_abbrev))
        
        game_id = result['game_id'] if result else None
        self._game_cache[cache_key] = game_id
        return game_id
    
    def _match_player(self, player_name: str) -> Optional[str]:
        """Match player name to our players table."""
        if player_name in self._player_cache:
            return self._player_cache[player_name]
        
        result = self.db.fetch_one(
            "SELECT player_id FROM players WHERE full_name = ?",
            (player_name,)
        )
        
        if result:
            self._player_cache[player_name] = result['player_id']
            return result['player_id']
        
        # Try fuzzy match
        parts = player_name.split()
        if len(parts) >= 2:
            result = self.db.fetch_one("""
                SELECT player_id FROM players 
                WHERE full_name LIKE ? LIMIT 1
            """, (f"%{parts[-1]}%",))
            
            if result:
                self._player_cache[player_name] = result['player_id']
                return result['player_id']
        
        self._player_cache[player_name] = None
        return None
    
    def _insert_line(self, cursor, line: Dict) -> bool:
        """Insert a betting line."""
        try:
            cursor.execute("""
                INSERT INTO betting_lines (
                    odds_api_event_id, game_id, season, week,
                    home_team, away_team, commence_time,
                    market_type, bookmaker, bookmaker_title,
                    player_name, player_id,
                    outcome_name, outcome_description, line_value,
                    odds_american, odds_decimal,
                    odds_api_last_update, snapshot_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                line.get('event_id'),
                line.get('game_id'),
                line['season'],
                line['week'],
                line.get('home_team'),
                line.get('away_team'),
                line.get('commence_time'),
                line['market_type'],
                line['bookmaker'],
                line.get('bookmaker_title'),
                line.get('player_name'),
                line.get('player_id'),
                line['outcome_name'],
                line.get('outcome_description'),
                line.get('line_value'),
                line.get('odds_american'),
                line.get('odds_decimal'),
                line.get('last_update'),
                datetime.now().isoformat(),
            ))
            self.stats['lines_inserted'] += 1
            return True
        except Exception as e:
            logger.debug(f"Insert error: {e}")
            return False
    
    def fetch_featured_odds(self, bookmakers: Optional[List[str]] = None) -> Dict:
        """Fetch moneyline, spreads, and totals."""
        logger.info("Fetching featured odds...")
        
        params = {
            'regions': ODDS_API_REGIONS,
            'markets': ','.join(ODDS_API_FEATURED_MARKETS),
            'oddsFormat': 'american',
        }
        
        if bookmakers:
            params['bookmakers'] = ','.join(bookmakers)
        
        data, headers = self._make_request(self.endpoints['odds'], params)
        
        if not data:
            return {'success': False, 'error': 'No data'}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for event in data:
                home_team = event.get('home_team', '')
                away_team = event.get('away_team', '')
                commence_time = event.get('commence_time', '')
                event_id = event.get('id', '')
                
                season, week = self._get_season_week(commence_time)
                game_id = self._match_game(home_team, away_team, season, week)
                
                for bookmaker in event.get('bookmakers', []):
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            odds_american = outcome.get('price')
                            
                            line = {
                                'event_id': event_id,
                                'game_id': game_id,
                                'season': season,
                                'week': week,
                                'home_team': home_team,
                                'away_team': away_team,
                                'commence_time': commence_time,
                                'market_type': market.get('key'),
                                'bookmaker': bookmaker.get('key'),
                                'bookmaker_title': bookmaker.get('title'),
                                'outcome_name': outcome.get('name'),
                                'line_value': outcome.get('point'),
                                'odds_american': odds_american,
                                'odds_decimal': american_odds_to_decimal(odds_american),
                                'last_update': bookmaker.get('last_update'),
                            }
                            
                            self._insert_line(cursor, line)
                
                self.stats['events_processed'] += 1
        
        logger.info(f"Featured odds: {self.stats['events_processed']} events, {self.stats['lines_inserted']} lines")
        
        return {
            'success': True,
            'events': self.stats['events_processed'],
            'lines': self.stats['lines_inserted'],
            'credits_remaining': self.stats['credits_remaining'],
        }
    
    def fetch_player_props(
        self,
        markets: Optional[List[str]] = None,
        bookmakers: Optional[List[str]] = None,
        max_events: Optional[int] = None
    ) -> Dict:
        """Fetch player prop odds."""
        logger.info("Fetching player props...")
        
        markets = markets or ODDS_API_PLAYER_PROP_MARKETS
        
        # Get events first
        events_data, _ = self._make_request(
            self.endpoints['events'],
            {'regions': ODDS_API_REGIONS}
        )
        
        if not events_data:
            return {'success': False, 'error': 'No events'}
        
        event_ids = [e['id'] for e in events_data]
        if max_events:
            event_ids = event_ids[:max_events]
        
        logger.info(f"Fetching props for {len(event_ids)} events")
        
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
                    home_team = data.get('home_team', '')
                    away_team = data.get('away_team', '')
                    commence_time = data.get('commence_time', '')
                    
                    season, week = self._get_season_week(commence_time)
                    game_id = self._match_game(home_team, away_team, season, week)
                    
                    for bookmaker in data.get('bookmakers', []):
                        for market in bookmaker.get('markets', []):
                            for outcome in market.get('outcomes', []):
                                player_name = outcome.get('description') or outcome.get('name', '')
                                player_id = self._match_player(player_name)
                                odds_american = outcome.get('price')
                                
                                line = {
                                    'event_id': event_id,
                                    'game_id': game_id,
                                    'season': season,
                                    'week': week,
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'commence_time': commence_time,
                                    'market_type': market.get('key'),
                                    'bookmaker': bookmaker.get('key'),
                                    'bookmaker_title': bookmaker.get('title'),
                                    'player_name': player_name,
                                    'player_id': player_id,
                                    'outcome_name': outcome.get('name'),
                                    'line_value': outcome.get('point'),
                                    'odds_american': odds_american,
                                    'odds_decimal': american_odds_to_decimal(odds_american),
                                    'last_update': bookmaker.get('last_update'),
                                }
                                
                                self._insert_line(cursor, line)
                    
                    self.stats['events_processed'] += 1
                
                time.sleep(0.1)  # Small delay
        
        logger.info(f"Player props: {self.stats['events_processed']} events, {self.stats['lines_inserted']} lines")
        
        return {
            'success': True,
            'events': self.stats['events_processed'],
            'lines': self.stats['lines_inserted'],
            'credits_remaining': self.stats['credits_remaining'],
        }
    
    def fetch_vegas_context(self) -> Dict:
        """
        Fetch odds and populate vegas_game_context table.
        This creates derived features for ML models.
        """
        logger.info("Building Vegas game context...")
        
        # First ensure we have featured odds
        result = self.fetch_featured_odds([ODDS_API_PREFERRED_BOOKMAKER])
        
        if not result['success']:
            return result
        
        # Now build context table
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get distinct games from betting lines
            games = self.db.fetch_all("""
                SELECT DISTINCT 
                    game_id, season, week, home_team, away_team,
                    odds_api_event_id
                FROM betting_lines
                WHERE game_id IS NOT NULL
                  AND bookmaker = ?
            """, (ODDS_API_PREFERRED_BOOKMAKER,))
            
            for game in games:
                game_id = game['game_id']
                
                # Get spread
                spread = self.db.fetch_one("""
                    SELECT line_value, odds_american
                    FROM betting_lines
                    WHERE game_id = ? AND market_type = 'spreads'
                      AND bookmaker = ? AND outcome_name = ?
                    ORDER BY snapshot_timestamp DESC LIMIT 1
                """, (game_id, ODDS_API_PREFERRED_BOOKMAKER, game['home_team']))
                
                # Get total
                total = self.db.fetch_one("""
                    SELECT line_value, odds_american
                    FROM betting_lines
                    WHERE game_id = ? AND market_type = 'totals'
                      AND bookmaker = ? AND outcome_name = 'Over'
                    ORDER BY snapshot_timestamp DESC LIMIT 1
                """, (game_id, ODDS_API_PREFERRED_BOOKMAKER))
                
                # Get moneylines
                home_ml = self.db.fetch_one("""
                    SELECT odds_american
                    FROM betting_lines
                    WHERE game_id = ? AND market_type = 'h2h'
                      AND bookmaker = ? AND outcome_name = ?
                    ORDER BY snapshot_timestamp DESC LIMIT 1
                """, (game_id, ODDS_API_PREFERRED_BOOKMAKER, game['home_team']))
                
                away_ml = self.db.fetch_one("""
                    SELECT odds_american
                    FROM betting_lines
                    WHERE game_id = ? AND market_type = 'h2h'
                      AND bookmaker = ? AND outcome_name = ?
                    ORDER BY snapshot_timestamp DESC LIMIT 1
                """, (game_id, ODDS_API_PREFERRED_BOOKMAKER, game['away_team']))
                
                # Calculate implied totals
                spread_line = spread['line_value'] if spread else None
                total_line = total['line_value'] if total else None
                
                home_implied = None
                away_implied = None
                if spread_line is not None and total_line is not None:
                    home_implied = (total_line - spread_line) / 2
                    away_implied = (total_line + spread_line) / 2
                
                # Insert context
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO vegas_game_context (
                            game_id, season, week, home_team, away_team,
                            spread_line, home_spread_odds,
                            total_line, over_odds,
                            home_ml_odds, away_ml_odds,
                            home_implied_total, away_implied_total,
                            bookmaker, snapshot_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_id, game['season'], game['week'],
                        game['home_team'], game['away_team'],
                        spread_line,
                        spread['odds_american'] if spread else None,
                        total_line,
                        total['odds_american'] if total else None,
                        home_ml['odds_american'] if home_ml else None,
                        away_ml['odds_american'] if away_ml else None,
                        home_implied, away_implied,
                        ODDS_API_PREFERRED_BOOKMAKER,
                        datetime.now().isoformat()
                    ))
                except Exception as e:
                    logger.debug(f"Vegas context insert error: {e}")
        
        logger.info("Vegas context built")
        return {'success': True}
    
    def weekly_pull(self, include_props: bool = True) -> Dict:
        """Run complete weekly pull."""
        logger.info("Starting weekly betting pull...")
        
        print(f"\n{'='*60}")
        print("Betting Lines Weekly Pull")
        print(f"{'='*60}\n")
        
        if not ODDS_API_KEY:
            print("ERROR: ODDS_API_KEY not set!")
            print("Set it with: export ODDS_API_KEY='your_key_here'")
            return {'success': False, 'error': 'No API key'}
        
        results = {'featured': None, 'props': None, 'context': None}
        
        # Featured odds
        print("1. Fetching featured odds (spreads, totals, moneylines)...")
        results['featured'] = self.fetch_featured_odds()
        print(f"   Ã¢Å“â€œ {results['featured'].get('lines', 0)} lines")
        
        # Player props
        if include_props:
            print("2. Fetching player props...")
            self.stats['lines_inserted'] = 0  # Reset for props
            self.stats['events_processed'] = 0
            results['props'] = self.fetch_player_props()
            print(f"   Ã¢Å“â€œ {results['props'].get('lines', 0)} lines")
        
        # Vegas context
        print("3. Building Vegas game context...")
        results['context'] = self.fetch_vegas_context()
        print("   Ã¢Å“â€œ Context built")
        
        total_lines = (
            (results['featured'].get('lines', 0) if results['featured'] else 0) +
            (results['props'].get('lines', 0) if results['props'] else 0)
        )
        
        print(f"\n{'='*60}")
        print(f"COMPLETE: {total_lines} total lines")
        print(f"Credits remaining: {self.stats['credits_remaining']}")
        print(f"{'='*60}\n")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch betting lines from The Odds API")
    parser.add_argument('--weekly', action='store_true', help='Full weekly pull')
    parser.add_argument('--featured', action='store_true', help='Featured only')
    parser.add_argument('--props', action='store_true', help='Props only')
    parser.add_argument('--no-props', action='store_true', help='Skip props in weekly')
    
    args = parser.parse_args()
    
    db = get_db()
    ingestion = BettingLinesIngestion(db)
    
    if args.featured:
        result = ingestion.fetch_featured_odds()
        print(f"Featured: {result}")
    elif args.props:
        result = ingestion.fetch_player_props()
        print(f"Props: {result}")
    else:
        result = ingestion.weekly_pull(include_props=not args.no_props)
    
    return 0


if __name__ == "__main__":
    exit(main())
