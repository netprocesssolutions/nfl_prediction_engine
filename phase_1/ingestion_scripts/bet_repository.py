"""
Bet Repository for NFL Fantasy Prediction Engine

This module provides CRUD operations and analysis functions for the
betting system, enabling:
- Creating bets from betting lines or manually
- Grading bets after games complete
- Analyzing performance by edge bucket, confidence, market, player
- Parlay support

Author: NFL Fantasy Prediction Engine Team
Phase: 1 Extension - Betting Integration
Version: 2.0
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import uuid

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger

# Import settings if available
try:
    from config.settings_odds_api import (
        calculate_edge_bucket,
        american_odds_to_decimal,
        american_odds_to_implied_probability,
    )
except ImportError:
    # Fallback implementations if settings not available
    def calculate_edge_bucket(expected_edge: float) -> str:
        abs_edge = abs(expected_edge) if expected_edge is not None else 0
        if abs_edge < 1: return "minimal"
        elif abs_edge < 3: return "low"
        elif abs_edge < 5: return "medium"
        elif abs_edge < 10: return "high"
        else: return "extreme"
    
    def american_odds_to_decimal(american_odds: int) -> float:
        if american_odds is None: return None
        if american_odds >= 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def american_odds_to_implied_probability(american_odds: int) -> float:
        if american_odds is None: return None
        if american_odds >= 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

# Initialize logger
logger = get_ingestion_logger("bet_repository")


class BetRepository:
    """
    Repository for managing bets with comprehensive tracking and analysis.
    
    Supports:
    - Creating bets from betting lines or manual input
    - Grading bets after games complete
    - Performance analysis by various dimensions
    - Parlay management
    """
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        """
        Initialize the repository.
        
        Args:
            db: Optional database connection. Uses global if not provided.
        """
        self.db = db or get_db()
    
    # =========================================================================
    # BET CREATION
    # =========================================================================
    
    def create_bet(
        self,
        season: int,
        week: int,
        market_type: str,
        stat_key: str,
        line_value: float,
        our_projection: float,
        operator: str = ">",
        confidence: Optional[float] = None,
        game_id: Optional[str] = None,
        player_id: Optional[str] = None,
        team_id: Optional[str] = None,
        player_name: Optional[str] = None,
        bookmaker: Optional[str] = None,
        odds_american: Optional[int] = None,
        stake: Optional[float] = None,
        betting_line_id: Optional[int] = None,
        is_parlay_leg: bool = False,
        parlay_group_id: Optional[str] = None,
    ) -> int:
        """
        Create a new bet record.
        
        Args:
            season: NFL season year
            week: NFL week number
            market_type: Type of market (player_prop, spread, moneyline, etc.)
            stat_key: Statistic being bet on (pass_yds, rush_yds, etc.)
            line_value: The sportsbook line
            our_projection: Our projected value
            operator: Comparison operator (>, <, >=, <=, =)
            confidence: Our confidence score (0.0-1.0)
            game_id: Optional game reference
            player_id: Optional player reference
            team_id: Optional team reference
            player_name: Player name for display
            bookmaker: Sportsbook name
            odds_american: American odds
            stake: Amount staked
            betting_line_id: Link to betting_lines table
            is_parlay_leg: Whether this is part of a parlay
            parlay_group_id: UUID linking parlay legs together
            
        Returns:
            bet_id of created record
        """
        # Calculate derived fields
        expected_edge = our_projection - line_value
        edge_bucket = calculate_edge_bucket(expected_edge)
        odds_decimal = american_odds_to_decimal(odds_american) if odds_american else None
        implied_probability = american_odds_to_implied_probability(odds_american) if odds_american else None
        potential_payout = stake * odds_decimal if stake and odds_decimal else None
        
        sql = """
            INSERT INTO bets_v2 (
                season, week, game_id, player_id, team_id,
                betting_line_id, market_type, stat_key, operator,
                bookmaker, player_name,
                line_value, odds_american, odds_decimal,
                our_projection, confidence, expected_edge, implied_probability,
                stake, potential_payout, is_parlay_leg, parlay_group_id,
                edge_bucket, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """
        
        params = (
            season, week, game_id, player_id, team_id,
            betting_line_id, market_type, stat_key, operator,
            bookmaker, player_name,
            line_value, odds_american, odds_decimal,
            our_projection, confidence, expected_edge, implied_probability,
            stake, potential_payout, 1 if is_parlay_leg else 0, parlay_group_id,
            edge_bucket,
        )
        
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, params)
            bet_id = cursor.lastrowid
        
        logger.info(
            f"Created bet {bet_id}: {stat_key} {operator} {line_value} (edge: {expected_edge:.1f})",
            event="bet_created"
        )
        
        return bet_id
    
    def create_bet_from_line(
        self,
        line_id: int,
        our_projection: float,
        confidence: Optional[float] = None,
        stake: Optional[float] = None,
        operator: str = ">",
    ) -> int:
        """
        Create a bet directly from a betting_lines record.
        
        Args:
            line_id: ID from betting_lines table
            our_projection: Our projected value
            confidence: Our confidence score
            stake: Amount to stake
            operator: Comparison operator
            
        Returns:
            bet_id of created record
        """
        # Fetch the betting line
        line = self.db.fetch_one(
            "SELECT * FROM betting_lines WHERE line_id = ?",
            (line_id,)
        )
        
        if not line:
            raise ValueError(f"Betting line {line_id} not found")
        
        return self.create_bet(
            season=line['season'],
            week=line['week'],
            market_type=line['market_type'],
            stat_key=line['market_type'],  # Will be mapped appropriately
            line_value=line['line_value'] or 0,
            our_projection=our_projection,
            operator=operator,
            confidence=confidence,
            game_id=line['game_id'],
            player_id=line['player_id'],
            player_name=line['player_name'],
            bookmaker=line['bookmaker'],
            odds_american=line['odds_american'],
            stake=stake,
            betting_line_id=line_id,
        )
    
    def create_parlay(
        self,
        bets: List[Dict],
        stake: float,
    ) -> Tuple[str, List[int]]:
        """
        Create a parlay bet with multiple legs.
        
        Args:
            bets: List of bet dictionaries with required fields
            stake: Total stake on the parlay
            
        Returns:
            Tuple of (parlay_group_id, list of bet_ids)
        """
        parlay_group_id = str(uuid.uuid4())
        bet_ids = []
        
        for i, bet in enumerate(bets):
            bet_id = self.create_bet(
                **bet,
                stake=stake if i == 0 else None,  # Only first leg has stake
                is_parlay_leg=True,
                parlay_group_id=parlay_group_id,
            )
            bet_ids.append(bet_id)
        
        logger.info(
            f"Created parlay {parlay_group_id} with {len(bet_ids)} legs",
            event="parlay_created"
        )
        
        return parlay_group_id, bet_ids
    
    # =========================================================================
    # BET GRADING
    # =========================================================================
    
    def grade_bet(
        self,
        bet_id: int,
        actual_value: float,
        force_outcome: Optional[str] = None,
    ) -> Dict:
        """
        Grade a bet after the game completes.
        
        Args:
            bet_id: ID of the bet to grade
            actual_value: The actual stat value achieved
            force_outcome: Optional override for outcome ('win', 'loss', 'push', 'void')
            
        Returns:
            Dictionary with grading results
        """
        # Fetch the bet
        bet = self.db.fetch_one(
            "SELECT * FROM bets_v2 WHERE bet_id = ?",
            (bet_id,)
        )
        
        if not bet:
            raise ValueError(f"Bet {bet_id} not found")
        
        # Determine outcome based on operator
        if force_outcome:
            outcome = force_outcome
        else:
            line_value = bet['line_value']
            operator = bet['operator']
            
            if operator == '>':
                outcome = 'win' if actual_value > line_value else 'loss'
            elif operator == '<':
                outcome = 'win' if actual_value < line_value else 'loss'
            elif operator == '>=':
                outcome = 'win' if actual_value >= line_value else 'loss'
            elif operator == '<=':
                outcome = 'win' if actual_value <= line_value else 'loss'
            elif operator == '=':
                outcome = 'win' if actual_value == line_value else 'loss'
            else:
                outcome = 'win' if actual_value > line_value else 'loss'
            
            # Handle push (typically for exact matches on spreads)
            if actual_value == line_value and operator in ('>', '<'):
                outcome = 'push'
        
        # Calculate metrics
        edge_realized = actual_value - bet['line_value']
        projection_error = actual_value - bet['our_projection']
        was_correct = 1 if outcome == 'win' else 0
        
        # Calculate profit/loss
        if outcome == 'win' and bet['stake'] and bet['odds_decimal']:
            profit_loss = bet['stake'] * (bet['odds_decimal'] - 1)
        elif outcome == 'loss' and bet['stake']:
            profit_loss = -bet['stake']
        elif outcome == 'push' and bet['stake']:
            profit_loss = 0
        else:
            profit_loss = None
        
        # Update the bet
        sql = """
            UPDATE bets_v2
            SET actual_value = ?,
                outcome = ?,
                profit_loss = ?,
                edge_realized = ?,
                projection_error = ?,
                was_correct = ?,
                graded_at = ?,
                updated_at = ?
            WHERE bet_id = ?
        """
        
        now = datetime.now().isoformat()
        params = (
            actual_value, outcome, profit_loss,
            edge_realized, projection_error, was_correct,
            now, now, bet_id
        )
        
        with self.db.get_cursor() as cursor:
            cursor.execute(sql, params)
        
        result = {
            'bet_id': bet_id,
            'outcome': outcome,
            'actual_value': actual_value,
            'line_value': bet['line_value'],
            'our_projection': bet['our_projection'],
            'edge_realized': edge_realized,
            'projection_error': projection_error,
            'profit_loss': profit_loss,
        }
        
        logger.info(
            f"Graded bet {bet_id}: {outcome} (actual: {actual_value}, line: {bet['line_value']})",
            event="bet_graded"
        )
        
        return result
    
    def auto_grade_bets_from_stats(
        self,
        season: int,
        week: int,
    ) -> Dict:
        """
        Automatically grade pending bets using player_game_stats data.
        
        Args:
            season: NFL season
            week: NFL week
            
        Returns:
            Dictionary with grading summary
        """
        # Mapping from stat_key to player_game_stats columns
        stat_column_map = {
            'pass_yds': 'pass_yards',
            'rush_yds': 'rush_yards',
            'rec_yds': 'rec_yards',
            'receptions': 'receptions',
            'pass_tds': 'pass_tds',
            'rush_tds': 'rush_tds',
            'rec_tds': 'rec_tds',
            'targets': 'targets',
            'carries': 'carries',
            'completions': 'completions',
            'pass_attempts': 'pass_attempts',
        }
        
        # Get pending bets for this week
        pending_bets = self.db.fetch_all("""
            SELECT bet_id, player_id, stat_key
            FROM bets_v2
            WHERE season = ? AND week = ? AND outcome = 'pending' AND player_id IS NOT NULL
        """, (season, week))
        
        graded = 0
        skipped = 0
        errors = []
        
        for bet in pending_bets:
            stat_key = bet['stat_key']
            player_id = bet['player_id']
            
            if stat_key not in stat_column_map:
                skipped += 1
                continue
            
            column = stat_column_map[stat_key]
            
            # Fetch actual stats
            stats = self.db.fetch_one(f"""
                SELECT {column} as actual_value
                FROM player_game_stats
                WHERE player_id = ? AND season = ? AND week = ?
            """, (player_id, season, week))
            
            if stats and stats['actual_value'] is not None:
                try:
                    self.grade_bet(bet['bet_id'], stats['actual_value'])
                    graded += 1
                except Exception as e:
                    errors.append(str(e))
            else:
                skipped += 1
        
        result = {
            'season': season,
            'week': week,
            'graded': graded,
            'skipped': skipped,
            'errors': errors,
        }
        
        logger.info(
            f"Auto-graded {graded} bets for {season} week {week}",
            event="auto_grade_complete"
        )
        
        return result
    
    # =========================================================================
    # BET RETRIEVAL
    # =========================================================================
    
    def get_bet(self, bet_id: int) -> Optional[Dict]:
        """Get a single bet by ID."""
        result = self.db.fetch_one(
            "SELECT * FROM bets_v2 WHERE bet_id = ?",
            (bet_id,)
        )
        return dict(result) if result else None
    
    def get_pending_bets(
        self,
        season: Optional[int] = None,
        week: Optional[int] = None,
    ) -> List[Dict]:
        """Get all pending bets, optionally filtered by season/week."""
        sql = "SELECT * FROM bets_v2 WHERE outcome = 'pending'"
        params = []
        
        if season:
            sql += " AND season = ?"
            params.append(season)
        
        if week:
            sql += " AND week = ?"
            params.append(week)
        
        sql += " ORDER BY created_at DESC"
        
        results = self.db.fetch_all(sql, tuple(params))
        return [dict(r) for r in results]
    
    def get_bets_by_parlay(self, parlay_group_id: str) -> List[Dict]:
        """Get all bets in a parlay."""
        results = self.db.fetch_all(
            "SELECT * FROM bets_v2 WHERE parlay_group_id = ? ORDER BY bet_id",
            (parlay_group_id,)
        )
        return [dict(r) for r in results]
    
    # =========================================================================
    # PERFORMANCE ANALYSIS
    # =========================================================================
    
    def get_performance_by_edge_bucket(
        self,
        season: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get win rate and ROI by edge bucket.
        
        Returns:
            List of dictionaries with bucket performance metrics
        """
        sql = """
            SELECT 
                edge_bucket,
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
                ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN was_correct ELSE NULL END) * 100, 1) as win_rate,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_profit_loss,
                ROUND(AVG(expected_edge), 2) as avg_expected_edge,
                ROUND(AVG(edge_realized), 2) as avg_edge_realized
            FROM bets_v2
            WHERE outcome != 'pending'
        """
        params = []
        
        if season:
            sql += " AND season = ?"
            params.append(season)
        
        sql += " GROUP BY edge_bucket ORDER BY avg_expected_edge"
        
        results = self.db.fetch_all(sql, tuple(params))
        return [dict(r) for r in results]
    
    def get_performance_by_confidence(
        self,
        bucket_size: float = 0.1,
        season: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get win rate by confidence bucket.
        
        Args:
            bucket_size: Size of confidence buckets (default 0.1 = 10%)
            season: Optional season filter
            
        Returns:
            List of dictionaries with confidence bucket performance
        """
        sql = f"""
            SELECT 
                CAST(confidence / {bucket_size} AS INTEGER) * {bucket_size} as confidence_bucket,
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN was_correct ELSE NULL END) * 100, 1) as win_rate,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_profit_loss
            FROM bets_v2
            WHERE outcome != 'pending' AND confidence IS NOT NULL
        """
        params = []
        
        if season:
            sql += " AND season = ?"
            params.append(season)
        
        sql += " GROUP BY confidence_bucket ORDER BY confidence_bucket"
        
        results = self.db.fetch_all(sql, tuple(params))
        return [dict(r) for r in results]
    
    def get_performance_by_market(
        self,
        season: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get win rate by market type and stat key.
        
        Returns:
            List of dictionaries with market performance metrics
        """
        sql = """
            SELECT 
                market_type,
                stat_key,
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN was_correct ELSE NULL END) * 100, 1) as win_rate,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_profit_loss,
                ROUND(AVG(ABS(projection_error)), 2) as avg_abs_error
            FROM bets_v2
            WHERE outcome != 'pending'
        """
        params = []
        
        if season:
            sql += " AND season = ?"
            params.append(season)
        
        sql += " GROUP BY market_type, stat_key ORDER BY total_bets DESC"
        
        results = self.db.fetch_all(sql, tuple(params))
        return [dict(r) for r in results]
    
    def get_performance_by_player(
        self,
        min_bets: int = 3,
        season: Optional[int] = None,
    ) -> List[Dict]:
        """
        Get win rate by player.
        
        Args:
            min_bets: Minimum number of bets to include player
            season: Optional season filter
            
        Returns:
            List of dictionaries with player performance metrics
        """
        sql = """
            SELECT 
                player_id,
                player_name,
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN was_correct ELSE NULL END) * 100, 1) as win_rate,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_profit_loss,
                ROUND(AVG(projection_error), 2) as avg_projection_error
            FROM bets_v2
            WHERE outcome != 'pending' AND player_id IS NOT NULL
        """
        params = []
        
        if season:
            sql += " AND season = ?"
            params.append(season)
        
        sql += f" GROUP BY player_id, player_name HAVING COUNT(*) >= {min_bets}"
        sql += " ORDER BY win_rate DESC"
        
        results = self.db.fetch_all(sql, tuple(params))
        return [dict(r) for r in results]
    
    def get_overall_stats(
        self,
        season: Optional[int] = None,
    ) -> Dict:
        """
        Get overall betting statistics.
        
        Returns:
            Dictionary with aggregate stats
        """
        sql = """
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN outcome = 'push' THEN 1 ELSE 0 END) as pushes,
                SUM(CASE WHEN outcome = 'pending' THEN 1 ELSE 0 END) as pending,
                ROUND(AVG(CASE WHEN outcome IN ('win', 'loss') THEN was_correct ELSE NULL END) * 100, 1) as win_rate,
                ROUND(SUM(COALESCE(stake, 0)), 2) as total_staked,
                ROUND(SUM(COALESCE(profit_loss, 0)), 2) as total_profit_loss
            FROM bets_v2
        """
        params = []
        
        if season:
            sql += " WHERE season = ?"
            params.append(season)
        
        result = self.db.fetch_one(sql, tuple(params))
        stats = dict(result) if result else {}
        
        # Calculate ROI
        if stats.get('total_staked') and stats['total_staked'] > 0:
            stats['roi'] = round(
                (stats.get('total_profit_loss', 0) / stats['total_staked']) * 100, 1
            )
        else:
            stats['roi'] = 0
        
        return stats
    
    def get_edge_analysis(
        self,
        season: Optional[int] = None,
    ) -> Dict:
        """
        Get comprehensive edge analysis to identify optimal betting thresholds.
        
        Returns:
            Dictionary with edge analysis insights
        """
        # Get performance by edge bucket
        edge_buckets = self.get_performance_by_edge_bucket(season)
        
        # Find optimal edge threshold (highest win rate with sufficient sample)
        optimal_bucket = None
        for bucket in edge_buckets:
            if bucket['total_bets'] >= 10 and bucket['win_rate'] and bucket['win_rate'] > 55:
                if not optimal_bucket or bucket['win_rate'] > optimal_bucket['win_rate']:
                    optimal_bucket = bucket
        
        # Calculate cumulative stats for edge >= threshold
        cumulative = []
        for i, bucket in enumerate(edge_buckets):
            remaining = edge_buckets[i:]
            total_bets = sum(b['total_bets'] for b in remaining)
            total_wins = sum(b['wins'] for b in remaining)
            total_pnl = sum(b['total_profit_loss'] or 0 for b in remaining)
            
            if total_bets > 0:
                cumulative.append({
                    'min_edge_bucket': bucket['edge_bucket'],
                    'total_bets': total_bets,
                    'wins': total_wins,
                    'win_rate': round((total_wins / total_bets) * 100, 1) if total_bets > 0 else 0,
                    'total_profit_loss': round(total_pnl, 2),
                })
        
        return {
            'by_bucket': edge_buckets,
            'cumulative': cumulative,
            'optimal_bucket': optimal_bucket,
            'recommendation': f"Focus on {optimal_bucket['edge_bucket']} edge bets or higher" if optimal_bucket else "Insufficient data for recommendation",
        }


# =============================================================================
# Module-level convenience function
# =============================================================================

def get_bet_repository() -> BetRepository:
    """Get a BetRepository instance."""
    return BetRepository(get_db())


if __name__ == "__main__":
    # Quick test
    print("Testing BetRepository...")
    
    repo = get_bet_repository()
    
    # Get overall stats
    stats = repo.get_overall_stats()
    print(f"\nOverall Stats: {stats}")
    
    # Get edge analysis
    analysis = repo.get_edge_analysis()
    print(f"\nEdge Buckets: {len(analysis['by_bucket'])} buckets")
    print(f"Recommendation: {analysis['recommendation']}")
