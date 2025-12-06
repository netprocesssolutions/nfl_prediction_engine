# NFL Fantasy Predictor - Fixed Files (December 2025)

## THE ROOT PROBLEM

**nflreadpy returns POLARS DataFrames, not Pandas DataFrames!**

All the errors you saw were because the code was calling Pandas methods (`.iterrows()`, `.rename(columns=...)`) on Polars DataFrames.

## WHAT WAS BROKEN

### 1. ingest_games.py
- **Error**: `KeyError: 'schedule_regular'`
- **Cause**: Sleeper API doesn't have a schedule endpoint. The code referenced `SLEEPER_ENDPOINTS["schedule_regular"]` which doesn't exist.
- **Fix**: Now uses `nflreadpy.load_schedules()` with proper Polars-to-Pandas conversion.

### 2. ingest_nflverse.py
- **Error**: `'DataFrame' object has no attribute 'iterrows'`
- **Cause**: nflreadpy returns Polars DataFrames, not Pandas. Polars doesn't have `.iterrows()`.
- **Fix**: Added `polars_to_pandas()` conversion function. All nflreadpy calls now convert to Pandas before processing.

### 3. ingest_stats_defense.py  
- **Error**: `IngestionLogger.warning() takes 2 positional arguments but 4 were given`
- **Cause**: Code used `logger.warning("message %s", arg)` but IngestionLogger expects `logger.warning(f"message {arg}")`
- **Error**: All 32 teams skipped with "unknown team_id"
- **Cause**: CSV has team nicknames ("Browns", "Lions") but code expected abbreviations ("CLE", "DET")
- **Fix**: Changed to f-string format for logger calls. Added `TEAM_NICKNAME_MAP` to convert nicknames to abbreviations.

### 4. run_master_pipeline.py
- **Error**: `OffensiveStatsIngestion.run() got an unexpected keyword argument 'season'`
- **Cause**: Method expects `seasons=` (plural, a list), not `season=` (singular)
- **Error**: `NFLVerseIngestion has no method 'ingest_all'`
- **Cause**: Method is called `run_full_ingestion()`, not `ingest_all()`
- **Fix**: Fixed all method calls to use correct signatures.

## HOW TO INSTALL THE FIXES

Replace these files in your `Phase 1/ingestion_scripts/` directory:

```
Phase 1/ingestion_scripts/
├── run_master_pipeline.py  ← REPLACE
├── ingest_games.py         ← REPLACE  
├── ingest_nflverse.py      ← REPLACE
└── ingest_stats_defense.py ← REPLACE
```

## EXPECTED RESULTS AFTER FIX

| Step | Before | After |
|------|--------|-------|
| Games | 0 (KeyError) | ~850 games (3 seasons × ~285 games) |
| Offensive Stats | 0 (wrong param) | ~50,000+ player-game stats |
| Defensive Stats | 1,601 (32 teams skipped) | 1,601 + 32 team season stats |
| NFLverse | 0-51,117 (errors) | 50,000+ rows (weekly stats, NGS, rosters, etc.) |
| Weather | 0 (depends on games) | ~16 games/week |

## KEY TECHNICAL NOTES

### Polars to Pandas Conversion
```python
def polars_to_pandas(df):
    """Convert Polars DataFrame to Pandas."""
    if df is None:
        return None
    if hasattr(df, 'iterrows'):  # Already Pandas
        return df
    try:
        return df.to_pandas()
    except Exception as e:
        return None
```

### Team Nickname Mapping
```python
TEAM_NICKNAME_MAP = {
    "BROWNS": "CLE",
    "LIONS": "DET", 
    "GIANTS": "NYG",
    # ... all 32 teams
}
```

### Correct Method Calls
```python
# WRONG:
ingestion.run(season=season)
nfl.ingest_all(seasons)

# CORRECT:
ingestion.run(seasons=seasons)  # plural, list
nfl.run_full_ingestion(seasons)
```

## RUN THE PIPELINE

After replacing the files:

```bash
cd "C:\Users\jlukt\FootballPredictor\Phase 1\ingestion_scripts"
python run_master_pipeline.py --skip-pbp
```

The `--skip-pbp` flag skips the large play-by-play dataset for faster testing.
