# NFL Fantasy Predictor - Fixed Files v2.3 (December 2025)

## PROBLEMS FIXED

### 1. Games Table Schema Mismatch (NEW!)
- **Error**: `table games has no column named home_score`
- **Cause**: My previous fix tried to insert `home_score`, `away_score`, `game_datetime`, `updated_at` columns that DON'T EXIST in your schema
- **Fix**: Now only inserts columns that actually exist: `game_id`, `season`, `week`, `home_team_id`, `away_team_id`, `datetime`, `stadium`, `created_at`

### 2. Polars vs Pandas DataFrames
- **Error**: `'DataFrame' object has no attribute 'iterrows'`
- **Cause**: nflreadpy returns Polars DataFrames, not Pandas
- **Fix**: Added `polars_to_pandas()` conversion function

### 3. Team Nickname Mapping  
- **Error**: All 32 teams skipped with "unknown team_id"
- **Cause**: CSV has nicknames ("Browns") but code expected abbreviations ("CLE")
- **Fix**: Added `TEAM_NICKNAME_MAP` dictionary

### 4. Logger Format
- **Error**: `IngestionLogger.warning() takes 2 positional arguments but 4 were given`
- **Fix**: Changed to f-string format

### 5. Method Call Signatures
- **Error**: `run() got an unexpected keyword argument 'season'`
- **Fix**: Changed to `seasons=` (plural, list) and correct method names

## DATABASE LOCATION

Based on your settings.py, the database should be at:
```
C:\Users\jlukt\FootballPredictor\Phase 1\database\nfl_data.db
```

The path is derived from:
```python
PROJECT_ROOT = Path(__file__).parent.parent  # Phase 1/
DATABASE_DIR = PROJECT_ROOT / "database"     # Phase 1/database/
DATABASE_PATH = DATABASE_DIR / "nfl_data.db" # Phase 1/database/nfl_data.db
```

If the `database` folder doesn't exist, the code should create it automatically.

**To verify your database location, run:**
```python
from utils.database import get_db
db = get_db()
print(f"Database path: {db.db_path}")
print(f"Exists: {db.db_path.exists()}")
```

## HOW TO INSTALL THE FIXES

Replace these files in your `Phase 1/ingestion_scripts/` directory:

```
Phase 1/ingestion_scripts/
├── run_master_pipeline.py  ← REPLACE
├── ingest_games.py         ← REPLACE (v2.3 - schema fix)
├── ingest_nflverse.py      ← REPLACE
└── ingest_stats_defense.py ← REPLACE
```

## RUN THE PIPELINE

```bash
cd "C:\Users\jlukt\FootballPredictor\Phase 1\ingestion_scripts"
python run_master_pipeline.py --skip-pbp
```

## EXPECTED RESULTS AFTER FIX

| Step | Before | After |
|------|--------|-------|
| Games | 0 (schema error) | ~850+ games |
| Offensive Stats | 0 (no games) | ~50,000+ |
| Team Defense Stats | 0 (name mismatch) | 32 |
| NFLverse | 53,999 ✓ | 53,999+ |

## ACTUAL GAMES TABLE SCHEMA

For reference, your games table has these columns ONLY:
- `game_id` TEXT PRIMARY KEY
- `season` INTEGER NOT NULL
- `week` INTEGER NOT NULL
- `home_team_id` TEXT NOT NULL
- `away_team_id` TEXT NOT NULL
- `datetime` TEXT
- `stadium` TEXT
- `weather_json` TEXT
- `created_at` TEXT

There is NO:
- ~~home_score~~
- ~~away_score~~
- ~~game_datetime~~ (it's just `datetime`)
- ~~updated_at~~
