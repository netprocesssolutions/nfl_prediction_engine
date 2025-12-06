# phase_2/schema.py

from dataclasses import dataclass, field
from typing import List


@dataclass
class PlayerGameFeaturesSchema:
    """
    Canonical schema for the player_game_features table.

    We separate the schema into groups:
    - identifiers
    - game context
    - labels (targets for training)
    - feature groups: usage_*, eff_*, form_*, team_*, opp_*, def_*, arch_*, etc.

    As we build each feature module, we'll extend the FEATURE_COLUMNS list.
    """

    # --- Identifiers ---
    identifiers: List[str] = field(
        default_factory=lambda: [
            "season",
            "week",
            "game_id",
            "player_id",
            "player_name",
            "position",
            "team",
            "opponent",
            "home_team",
            "away_team",
            "is_home",
            "is_playoff",
        ]
    )

    # --- Labels (fantasy-relevant outcomes) ---
    # These will be populated from existing stat tables (Phase 1).
    labels: List[str] = field(
        default_factory=lambda: [
            # Receiving
            "label_targets",
            "label_receptions",
            "label_rec_yards",
            "label_rec_tds",
            # Rushing
            "label_carries",
            "label_rush_yards",
            "label_rush_tds",
            # Passing
            "label_pass_attempts",
            "label_pass_completions",
            "label_pass_yards",
            "label_pass_tds",
            "label_interceptions",
            # Misc
            "label_fumbles",
            "label_two_pt_conversions",
        ]
    )

    # --- Feature columns (to be grown as Phase 2 develops) ---
    base_feature_columns: List[str] = field(
        default_factory=lambda: [
            # Usage – to be filled by features/usage.py
            # e.g., "usage_snap_share", "usage_route_share", "usage_target_share", ...
            # Efficiency – features/efficiency.py
            # Form – features/form.py
            # Team context – features/team_context.py
            # Opponent/team defense – features/team_defense.py
            # Defender matchup – features/defender_matchup.py
            # Archetypes – features/archetypes.py
        ]
    )

    def all_columns(self) -> List[str]:
        """
        Return full ordered list of columns expected in player_game_features.
        """
        return self.identifiers + self.labels + self.base_feature_columns


# Global schema instance we can import everywhere
PLAYER_GAME_FEATURES_SCHEMA = PlayerGameFeaturesSchema()
