# phase_3/predictors/baseline.py
"""
Baseline Predictor - Phase 3

A deterministic, rule-based predictor that combines:
1. Base player expectation (long-term averages)
2. Player form adjustment (recent performance weighted)
3. Team context adjustment (team tendencies)
4. Opposing defense adjustment (matchup difficulty)
5. Stability constraints (shrinkage, caps)

This predictor serves as:
- A standalone interpretable prediction system
- A fallback when ML models are unstable
- A base learner for the ensemble meta-model (Phase 6)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..db import read_sql, get_connection


@dataclass
class PredictionConfig:
    """Configuration for baseline predictions."""

    # Form adjustment weights (exponential decay for last 3 games)
    form_weights: Tuple[float, float, float] = (0.55, 0.30, 0.15)

    # Form influence by stat type (how much recent form affects prediction)
    form_influence: Dict[str, float] = None

    # Opponent defense influence
    defense_season_weight: float = 0.35
    defense_form_weight: float = 0.45

    # Stability shrinkage (blend prediction with long-term average)
    stability_shrinkage: float = 0.15
    rookie_shrinkage: float = 0.30

    # Minimum games for reliable prediction
    min_games_reliable: int = 3

    def __post_init__(self):
        if self.form_influence is None:
            self.form_influence = {
                'targets': 0.65,
                'receptions': 0.60,
                'rec_yards': 0.55,
                'rec_tds': 0.40,
                'carries': 0.60,
                'rush_yards': 0.50,
                'rush_tds': 0.40,
                'pass_attempts': 0.50,
                'completions': 0.50,
                'pass_yards': 0.45,
                'pass_tds': 0.35,
                'interceptions': 0.30,
            }


class BaselinePredictor:
    """
    Rule-based baseline predictor for fantasy football stats.

    Produces predictions for:
    - WR/TE: targets, receptions, rec_yards, rec_tds
    - RB: carries, rush_yards, rush_tds + receiving stats
    - QB: pass_attempts, completions, pass_yards, pass_tds, interceptions
    """

    VERSION = "3.0.0"

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self._league_averages: Optional[Dict] = None
        self._position_averages: Optional[Dict] = None

    def _load_league_averages(self, season: int) -> Dict[str, float]:
        """Load league-wide averages for normalization."""
        if self._league_averages is not None:
            return self._league_averages

        query = """
            SELECT
                AVG(label_targets) as avg_targets,
                AVG(label_receptions) as avg_receptions,
                AVG(label_rec_yards) as avg_rec_yards,
                AVG(label_rec_tds) as avg_rec_tds,
                AVG(label_carries) as avg_carries,
                AVG(label_rush_yards) as avg_rush_yards,
                AVG(label_rush_tds) as avg_rush_tds,
                AVG(label_pass_attempts) as avg_pass_attempts,
                AVG(label_pass_completions) as avg_completions,
                AVG(label_pass_yards) as avg_pass_yards,
                AVG(label_pass_tds) as avg_pass_tds,
                AVG(label_interceptions) as avg_interceptions
            FROM player_game_features
            WHERE season < ?
        """
        df = read_sql(query, [season])
        self._league_averages = df.iloc[0].to_dict()
        return self._league_averages

    def _load_position_averages(self, season: int) -> Dict[str, Dict[str, float]]:
        """Load position-specific averages."""
        if self._position_averages is not None:
            return self._position_averages

        query = """
            SELECT
                position,
                AVG(label_targets) as avg_targets,
                AVG(label_receptions) as avg_receptions,
                AVG(label_rec_yards) as avg_rec_yards,
                AVG(label_rec_tds) as avg_rec_tds,
                AVG(label_carries) as avg_carries,
                AVG(label_rush_yards) as avg_rush_yards,
                AVG(label_rush_tds) as avg_rush_tds,
                AVG(label_pass_attempts) as avg_pass_attempts,
                AVG(label_pass_completions) as avg_completions,
                AVG(label_pass_yards) as avg_pass_yards,
                AVG(label_pass_tds) as avg_pass_tds,
                AVG(label_interceptions) as avg_interceptions
            FROM player_game_features
            WHERE season < ?
            GROUP BY position
        """
        df = read_sql(query, [season])
        self._position_averages = {}
        for _, row in df.iterrows():
            pos = row['position']
            self._position_averages[pos] = row.drop('position').to_dict()
        return self._position_averages

    def _get_player_baseline(self, row: pd.Series, stat: str) -> float:
        """
        Get base player expectation for a stat.

        Uses season-to-date average if available, otherwise position average.
        """
        # Map stat to actual feature column names
        stat_to_feature = {
            'targets': 'usage_targets_season_to_date',
            'carries': 'usage_carries_season_to_date',
            'receptions': 'usage_targets_season_to_date',  # Use targets as proxy
            'rec_yards': None,  # Not directly available
            'rush_yards': None,  # Not directly available
            'pass_attempts': None,
            'completions': None,
            'pass_yards': None,
        }

        feature_col = stat_to_feature.get(stat)
        if feature_col and feature_col in row.index and pd.notna(row[feature_col]):
            val = row[feature_col]
            # For receptions, apply catch rate to targets
            if stat == 'receptions':
                val = val * 0.65  # Average catch rate
            return val

        # Fall back to position average
        position = row.get('position', 'WR')
        pos_avgs = self._position_averages.get(position, {})
        return pos_avgs.get(f'avg_{stat}', 0.0)

    def _get_form_adjustment(self, row: pd.Series, stat: str) -> float:
        """
        Calculate player form adjustment based on recent performance.

        Uses exponentially weighted average of last 1 and 3 games vs season average.
        """
        # Map stat to actual feature column names
        stat_to_cols = {
            'targets': ('usage_targets_last1', 'usage_targets_last3', 'usage_targets_season_to_date'),
            'carries': ('usage_carries_last1', 'usage_carries_last3', 'usage_carries_season_to_date'),
            'receptions': ('usage_targets_last1', 'usage_targets_last3', 'usage_targets_season_to_date'),
        }

        cols = stat_to_cols.get(stat)
        if not cols:
            return 0.0

        last1_col, last3_col, std_col = cols

        # Get values (default to NaN if missing)
        last1 = row.get(last1_col, np.nan)
        last3 = row.get(last3_col, np.nan)
        season_avg = row.get(std_col, np.nan)

        if pd.isna(season_avg) or season_avg == 0:
            return 0.0

        # Calculate form delta
        if pd.notna(last3):
            recent_form = last3
        elif pd.notna(last1):
            recent_form = last1
        else:
            return 0.0

        delta = recent_form - season_avg

        # Apply form influence coefficient
        influence = self.config.form_influence.get(stat, 0.5)
        return delta * influence

    def _get_defense_adjustment(self, row: pd.Series, stat: str, position: str) -> float:
        """
        Calculate opposing defense adjustment.

        Uses opponent defense features to adjust based on matchup difficulty.
        """
        # Map stat to opponent defense feature
        stat_to_oppdef = {
            'targets': f'oppdef_targets_allowed_to_{position.lower()}_last3',
            'receptions': f'oppdef_receptions_allowed_to_{position.lower()}_last3',
            'rec_yards': f'oppdef_yards_allowed_to_{position.lower()}_last3',
            'rec_tds': f'oppdef_tds_allowed_to_{position.lower()}_last3',
            'carries': 'oppdef_carries_allowed_to_rb_last3',
            'rush_yards': 'oppdef_yards_allowed_to_rb_last3',
            'rush_tds': 'oppdef_tds_allowed_to_rb_last3',
            'pass_yards': 'oppdef_pass_yards_allowed_last3',
            'pass_tds': 'oppdef_pass_tds_allowed_last3',
        }

        oppdef_col = stat_to_oppdef.get(stat)
        if not oppdef_col or oppdef_col not in row.index:
            return 0.0

        oppdef_val = row.get(oppdef_col, np.nan)
        if pd.isna(oppdef_val):
            return 0.0

        # Get league average for comparison
        league_avg = self._league_averages.get(f'avg_{stat}', 0)
        if league_avg == 0:
            return 0.0

        # Calculate adjustment: positive if defense allows more than average
        # Normalize by comparing to typical allowed rate
        adjustment = (oppdef_val - league_avg) * self.config.defense_form_weight
        return adjustment

    def _apply_stability(self, prediction: float, baseline: float,
                         is_rookie: bool = False) -> float:
        """
        Apply stability shrinkage toward baseline.

        Blends prediction with long-term average for stability.
        """
        shrinkage = self.config.rookie_shrinkage if is_rookie else self.config.stability_shrinkage
        return (1 - shrinkage) * prediction + shrinkage * baseline

    def _apply_constraints(self, predictions: Dict[str, float],
                          position: str) -> Dict[str, float]:
        """
        Apply logical constraints to predictions.

        Ensures:
        - receptions <= targets
        - completions <= pass_attempts
        - No negative predictions
        """
        result = predictions.copy()

        # Non-negative constraint
        for stat in result:
            result[stat] = max(0.0, result[stat])

        # Receptions can't exceed targets
        if 'receptions' in result and 'targets' in result:
            result['receptions'] = min(result['receptions'], result['targets'])

        # Completions can't exceed attempts
        if 'completions' in result and 'pass_attempts' in result:
            result['completions'] = min(result['completions'], result['pass_attempts'])

        # Position-specific caps
        if position == 'QB':
            result['targets'] = min(result.get('targets', 0), 5)  # QBs rarely targeted
            result['carries'] = min(result.get('carries', 0), 15)  # Cap QB rushes
        elif position in ('WR', 'TE'):
            result['carries'] = min(result.get('carries', 0), 5)  # WR/TE rarely carry
            result['pass_attempts'] = 0  # WR/TE don't pass
        elif position == 'RB':
            result['pass_attempts'] = 0  # RBs don't pass

        return result

    def predict_player(self, row: pd.Series) -> Dict[str, float]:
        """
        Generate predictions for a single player-game.

        Args:
            row: Series containing player_game_features for one player-game

        Returns:
            Dictionary of predicted stats
        """
        position = row.get('position', 'WR')

        # Determine which stats to predict based on position
        if position == 'QB':
            stats_to_predict = ['pass_attempts', 'completions', 'pass_yards',
                              'pass_tds', 'interceptions', 'carries',
                              'rush_yards', 'rush_tds']
        elif position == 'RB':
            stats_to_predict = ['carries', 'rush_yards', 'rush_tds',
                              'targets', 'receptions', 'rec_yards', 'rec_tds']
        else:  # WR, TE
            stats_to_predict = ['targets', 'receptions', 'rec_yards', 'rec_tds']

        predictions = {}

        for stat in stats_to_predict:
            # 1. Base player expectation
            base = self._get_player_baseline(row, stat)

            # 2. Form adjustment
            form_adj = self._get_form_adjustment(row, stat)

            # 3. Defense adjustment
            defense_adj = self._get_defense_adjustment(row, stat, position)

            # 4. Combine adjustments
            raw_prediction = base + form_adj + defense_adj

            # 5. Apply stability shrinkage
            prediction = self._apply_stability(raw_prediction, base)

            predictions[stat] = prediction

        # 6. Apply logical constraints
        predictions = self._apply_constraints(predictions, position)

        return predictions

    def predict_week(self, season: int, week: int) -> pd.DataFrame:
        """
        Generate predictions for all players in a given week.

        Args:
            season: Season year
            week: Week number

        Returns:
            DataFrame with predictions for all players
        """
        # Load league/position averages
        self._load_league_averages(season)
        self._load_position_averages(season)

        # Load features for the week
        query = """
            SELECT *
            FROM player_game_features
            WHERE season = ? AND week = ?
        """
        features_df = read_sql(query, [season, week])

        if features_df.empty:
            print(f"No features found for {season} week {week}")
            return pd.DataFrame()

        print(f"Generating predictions for {len(features_df)} players...")

        # Generate predictions for each player
        predictions = []
        for _, row in features_df.iterrows():
            player_preds = self.predict_player(row)

            # Add identifiers
            player_preds['player_id'] = row['player_id']
            player_preds['game_id'] = row['game_id']
            player_preds['season'] = season
            player_preds['week'] = week
            player_preds['player_name'] = row.get('player_name', '')
            player_preds['position'] = row.get('position', '')
            player_preds['team'] = row.get('team', '')

            predictions.append(player_preds)

        # Convert to DataFrame
        pred_df = pd.DataFrame(predictions)

        # Reorder columns
        id_cols = ['player_id', 'game_id', 'season', 'week',
                   'player_name', 'position', 'team']
        stat_cols = ['targets', 'receptions', 'rec_yards', 'rec_tds',
                    'carries', 'rush_yards', 'rush_tds',
                    'pass_attempts', 'completions', 'pass_yards',
                    'pass_tds', 'interceptions']

        ordered_cols = id_cols + [c for c in stat_cols if c in pred_df.columns]
        pred_df = pred_df[[c for c in ordered_cols if c in pred_df.columns]]

        return pred_df

    def predict_season(self, season: int, start_week: int = 1,
                       end_week: int = 18) -> pd.DataFrame:
        """
        Generate predictions for an entire season.

        Args:
            season: Season year
            start_week: First week to predict
            end_week: Last week to predict

        Returns:
            DataFrame with all predictions
        """
        all_predictions = []

        for week in range(start_week, end_week + 1):
            week_preds = self.predict_week(season, week)
            if not week_preds.empty:
                all_predictions.append(week_preds)

        if not all_predictions:
            return pd.DataFrame()

        return pd.concat(all_predictions, ignore_index=True)

    def save_predictions(self, predictions: pd.DataFrame,
                        table_name: str = "baseline_predictions") -> int:
        """
        Save predictions to the database.

        Args:
            predictions: DataFrame of predictions
            table_name: Name of the table to save to

        Returns:
            Number of rows saved
        """
        if predictions.empty:
            return 0

        # Add metadata
        predictions = predictions.copy()
        predictions['baseline_version'] = self.VERSION
        predictions['created_at'] = datetime.utcnow().isoformat()

        with get_connection(readonly=False) as conn:
            # Create table if not exists
            predictions.to_sql(table_name, conn, if_exists='replace', index=False)

        print(f"Saved {len(predictions)} predictions to {table_name}")
        return len(predictions)


def calculate_fantasy_points(row: pd.Series, scoring: str = 'ppr') -> float:
    """
    Calculate fantasy points from predicted stats.

    Args:
        row: Series with predicted stats
        scoring: 'ppr', 'half_ppr', or 'standard'

    Returns:
        Fantasy points
    """
    points = 0.0

    # Helper to safely get numeric value
    def safe_get(key, default=0):
        val = row.get(key, default)
        return val if pd.notna(val) else default

    # Passing
    points += safe_get('pass_yards') * 0.04  # 1 point per 25 yards
    points += safe_get('pass_tds') * 4
    points += safe_get('interceptions') * -2

    # Rushing
    points += safe_get('rush_yards') * 0.1  # 1 point per 10 yards
    points += safe_get('rush_tds') * 6

    # Receiving
    points += safe_get('rec_yards') * 0.1  # 1 point per 10 yards
    points += safe_get('rec_tds') * 6

    # Receptions (PPR scoring)
    if scoring == 'ppr':
        points += safe_get('receptions') * 1.0
    elif scoring == 'half_ppr':
        points += safe_get('receptions') * 0.5

    return points
