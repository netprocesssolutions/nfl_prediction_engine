# Save this as merge_databases.py and run it from your repo root
import sqlite3
import shutil
import os

# Paths (adjust if needed)
base_db = "phase_1/database/nfl_data (1).db"  # Has predictions + processed PBP
config_db = "phase_1/config/database/nfl_data.db"  # Has raw play_by_play
output_db = "phase_1/database/nfl_data_complete.db"

# Start with the 90MB database (has predictions + PBP features)
shutil.copy(base_db, output_db)
print(f"Created {output_db} from base")

merged = sqlite3.connect(output_db)
merged.execute(f"ATTACH DATABASE '{config_db}' AS config_db")

# Add the raw play_by_play table (132K rows)
print("Adding play_by_play table...")
merged.execute("""
    CREATE TABLE IF NOT EXISTS play_by_play AS 
    SELECT * FROM config_db.play_by_play
""")
merged.commit()
count = merged.execute("SELECT COUNT(*) FROM play_by_play").fetchone()[0]
print(f"Added play_by_play: {count:,} rows")

merged.execute("DETACH DATABASE config_db")
merged.execute("VACUUM")
merged.close()

print(f"\n✅ Done! Merged database: {output_db}")
print(f"Size: {os.path.getsize(output_db) / 1024 / 1024:.1f} MB")
