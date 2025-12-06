"""
Weather Data Ingestion using Open-Meteo API (FREE - No API Key Required)

Fetches weather conditions for game locations to use as features.
Weather significantly impacts:
- Passing game efficiency
- Total points scored
- Specific player performance

Open-Meteo is completely FREE with no rate limits for reasonable usage.

Author: NFL Fantasy Prediction Engine Team
Version: 1.0
"""

import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from utils.database import get_db, DatabaseConnection
from utils.logger import get_ingestion_logger
from config.settings import (
    WEATHER_API_BASE_URL,
    STADIUM_COORDINATES,
    DOME_STADIUMS,
    CURRENT_SEASON,
)

logger = get_ingestion_logger("ingest_weather")


class WeatherIngestion:
    """Ingest weather data from Open-Meteo API."""
    
    def __init__(self, db: Optional[DatabaseConnection] = None):
        self.db = db or get_db()
        self.stats = {
            'games_processed': 0,
            'weather_fetched': 0,
            'dome_games': 0,
            'errors': [],
        }
    
    def fetch_weather(
        self,
        lat: float,
        lon: float,
        game_datetime: datetime
    ) -> Optional[Dict]:
        """
        Fetch weather for a specific location and time.
        
        Args:
            lat: Latitude
            lon: Longitude
            game_datetime: Game date and time
            
        Returns:
            Dictionary with weather conditions
        """
        try:
            params = {
                'latitude': lat,
                'longitude': lon,
                'hourly': 'temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,wind_gusts_10m,wind_direction_10m,visibility',
                'temperature_unit': 'fahrenheit',
                'wind_speed_unit': 'mph',
                'precipitation_unit': 'inch',
                'start_date': game_datetime.strftime('%Y-%m-%d'),
                'end_date': game_datetime.strftime('%Y-%m-%d'),
                'timezone': 'America/New_York',
            }
            
            response = requests.get(WEATHER_API_BASE_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.warning(f"Weather API error: {response.status_code}")
                return None
            
            data = response.json()
            
            # Find the hour closest to game time
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            
            game_hour = game_datetime.hour
            target_time = game_datetime.strftime('%Y-%m-%dT%H:00')
            
            # Find matching hour index
            hour_idx = 0
            for i, t in enumerate(times):
                if t == target_time:
                    hour_idx = i
                    break
            
            weather = {
                'temperature_f': hourly.get('temperature_2m', [None])[hour_idx],
                'humidity_pct': hourly.get('relative_humidity_2m', [None])[hour_idx],
                'precipitation_in': hourly.get('precipitation', [None])[hour_idx],
                'wind_speed_mph': hourly.get('wind_speed_10m', [None])[hour_idx],
                'wind_gust_mph': hourly.get('wind_gusts_10m', [None])[hour_idx],
                'wind_direction': hourly.get('wind_direction_10m', [None])[hour_idx],
                'visibility_miles': (hourly.get('visibility', [None])[hour_idx] or 0) / 1609.34,  # meters to miles
            }
            
            self.stats['weather_fetched'] += 1
            return weather
            
        except Exception as e:
            logger.warning(f"Weather fetch error: {e}")
            self.stats['errors'].append(str(e))
            return None
    
    def calculate_weather_impact(self, weather: Dict, is_dome: bool) -> float:
        """
        Calculate weather impact score (0-100).
        
        Higher score = more negative impact on passing game.
        """
        if is_dome or weather is None:
            return 0.0
        
        score = 0.0
        
        # Temperature impact
        temp = weather.get('temperature_f', 70)
        if temp is not None:
            if temp < 32:
                score += 25  # Freezing
            elif temp < 40:
                score += 15
            elif temp < 50:
                score += 5
            elif temp > 90:
                score += 10  # Extreme heat
        
        # Wind impact (biggest factor)
        wind = weather.get('wind_speed_mph', 0) or 0
        if wind >= 20:
            score += 30
        elif wind >= 15:
            score += 20
        elif wind >= 10:
            score += 10
        
        # Wind gusts
        gusts = weather.get('wind_gust_mph', 0) or 0
        if gusts >= 30:
            score += 15
        elif gusts >= 25:
            score += 10
        
        # Precipitation
        precip = weather.get('precipitation_in', 0) or 0
        if precip >= 0.5:
            score += 25  # Heavy rain/snow
        elif precip >= 0.1:
            score += 15
        elif precip > 0:
            score += 5
        
        return min(score, 100)
    
    def ingest_for_week(self, season: int, week: int) -> Dict:
        """Ingest weather for all games in a specific week."""
        logger.info(f"Ingesting weather for {season} week {week}")
        
        # Get games for this week - try games table first
        games = self.db.fetch_all("""
            SELECT 
                g.game_id, g.home_team_id, g.datetime
            FROM games g
            WHERE g.season = ? AND g.week = ?
        """, (season, week))
        
        if not games:
            # Try getting from schedules table (NFLverse format)
            games = self.db.fetch_all("""
                SELECT 
                    game_id, home_team as home_team_id, 
                    gameday || 'T' || COALESCE(gametime, '13:00:00') as datetime
                FROM schedules
                WHERE season = ? AND week = ?
            """, (season, week))
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for game in games:
                game_id = game['game_id']
                home_team = game['home_team_id']
                
                # Check if dome - use home team to determine
                is_dome = home_team in DOME_STADIUMS
                
                if is_dome:
                    # Dome game - no weather impact
                    weather = None
                    weather_impact = 0.0
                    self.stats['dome_games'] += 1
                else:
                    # Get coordinates
                    coords = STADIUM_COORDINATES.get(home_team)
                    if not coords:
                        logger.warning(f"No coordinates for {home_team}")
                        continue
                    
                    # Parse game time
                    try:
                        game_dt = datetime.fromisoformat(game['datetime'].replace('Z', ''))
                    except:
                        game_dt = datetime.now()
                    
                    # Fetch weather
                    weather = self.fetch_weather(coords[0], coords[1], game_dt)
                    weather_impact = self.calculate_weather_impact(weather, False)
                
                # Insert into table
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO game_weather (
                            game_id, season, week,
                            stadium, roof_type,
                            temperature_f, humidity_pct,
                            wind_speed_mph, wind_gust_mph,
                            precipitation_in,
                            weather_impact_score,
                            data_source, forecast_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        game_id, season, week,
                        home_team,  # Use team as stadium identifier
                        'dome' if is_dome else 'outdoor',
                        weather.get('temperature_f') if weather else None,
                        weather.get('humidity_pct') if weather else None,
                        weather.get('wind_speed_mph') if weather else None,
                        weather.get('wind_gust_mph') if weather else None,
                        weather.get('precipitation_in') if weather else None,
                        weather_impact,
                        'open-meteo',
                        datetime.now().isoformat()
                    ))
                except Exception as e:
                    logger.debug(f"Weather insert error: {e}")
                
                self.stats['games_processed'] += 1
        
        logger.info(
            f"Weather ingested: {self.stats['games_processed']} games, "
            f"{self.stats['dome_games']} dome, {self.stats['weather_fetched']} outdoor"
        )
        
        return {
            'success': True,
            'games_processed': self.stats['games_processed'],
            'weather_fetched': self.stats['weather_fetched'],
            'dome_games': self.stats['dome_games'],
        }
    
    def ingest_current_week(self) -> Dict:
        """Ingest weather for current week's games."""
        from config.settings import CURRENT_SEASON
        
        # Try to detect current week from schedules
        result = self.db.fetch_one("""
            SELECT MAX(week) as current_week
            FROM schedules
            WHERE season = ? AND gameday <= date('now')
        """, (CURRENT_SEASON,))
        
        current_week = result['current_week'] if result and result['current_week'] else 1
        
        return self.ingest_for_week(CURRENT_SEASON, current_week)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch weather data for NFL games")
    parser.add_argument('--season', type=int, default=CURRENT_SEASON, help='Season')
    parser.add_argument('--week', type=int, help='Week number')
    parser.add_argument('--current', action='store_true', help='Current week')
    
    args = parser.parse_args()
    
    db = get_db()
    ingestion = WeatherIngestion(db)
    
    if args.current:
        result = ingestion.ingest_current_week()
    elif args.week:
        result = ingestion.ingest_for_week(args.season, args.week)
    else:
        parser.print_help()
        return 0
    
    print(f"Result: {result}")
    return 0


if __name__ == "__main__":
    exit(main())
