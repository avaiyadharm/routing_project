"""Main entry point: CLI interface for optimal routing."""

import argparse
import json
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from router import OptimalRouter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_coordinates(coord_str: str) -> tuple:
    """Parse 'lat,lon' string to tuple."""
    try:
        lat, lon = map(float, coord_str.split(','))
        return lat, lon
    except ValueError:
        raise ValueError(f"Invalid coordinates: {coord_str}. Use format: lat,lon")


def main():
    """CLI entry point for routing engine."""
    parser = argparse.ArgumentParser(
        description='ML-driven optimal traffic routing engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Route with defaults (noon, Monday, June, no weather)
  python main.py --source 39.745,-75.546 --dest 39.758,-75.532

  # Morning commute in rain
  python main.py --source 39.745,-75.546 --dest 39.758,-75.532 \\
                 --hour 8 --day 0 --month 6 --raining 1

  # Evening rush hour with festival
  python main.py --source 39.745,-75.546 --dest 39.758,-75.532 \\
                 --hour 18 --day 4 --festival 1
        """
    )

    # Required arguments
    parser.add_argument(
        '--source', required=True, type=str,
        help='Source coordinates (lat,lon)'
    )
    parser.add_argument(
        '--dest', required=True, type=str,
        help='Destination coordinates (lat,lon)'
    )

    # Optional temporal parameters
    parser.add_argument(
        '--hour', type=int, default=12,
        help='Hour of day for prediction (0-23, default: 12)'
    )
    parser.add_argument(
        '--day', type=int, default=0,
        help='Day of week (0=Monday, 6=Sunday, default: 0)'
    )
    parser.add_argument(
        '--month', type=int, default=6,
        help='Month (1-12, default: 6=June)'
    )

    # Environmental conditions
    parser.add_argument(
        '--raining', type=int, choices=[0, 1], default=0,
        help='Is it raining? (0=no, 1=yes, default: 0)'
    )
    parser.add_argument(
        '--festival', type=int, choices=[0, 1], default=0,
        help='Is there a festival? (0=no, 1=yes, default: 0)'
    )

    # Output format
    parser.add_argument(
        '--json', action='store_true',
        help='Output as JSON'
    )

    args = parser.parse_args()

    # Validate arguments
    if not (0 <= args.hour <= 23):
        parser.error("Hour must be 0-23")
    if not (0 <= args.day <= 6):
        parser.error("Day must be 0-6")
    if not (1 <= args.month <= 12):
        parser.error("Month must be 1-12")

    try:
        source_lat, source_lon = parse_coordinates(args.source)
        dest_lat, dest_lon = parse_coordinates(args.dest)
    except ValueError as e:
        parser.error(str(e))

    logger.info("🚀 ML-driven Traffic Routing Engine")
    logger.info("=" * 60)

    try:
        # Initialize router
        router = OptimalRouter()

        # Find route
        result = router.find_optimal_route(
            start_lat=source_lat,
            start_lon=source_lon,
            end_lat=dest_lat,
            end_lon=dest_lon,
            departure_hour=args.hour,
            departure_day=args.day,
            departure_month=args.month,
            is_raining=args.raining,
            is_festival_zone=args.festival
        )

        # Output results
        if args.json:
            # JSON output
            output = {
                'status': 'success',
                'source': {'lat': source_lat, 'lon': source_lon},
                'destination': {'lat': dest_lat, 'lon': dest_lon},
                'conditions': {
                    'hour': args.hour,
                    'day_of_week': args.day,
                    'month': args.month,
                    'is_raining': bool(args.raining),
                    'is_festival_zone': bool(args.festival)
                },
                'route': {
                    'waypoints': [{'lat': lat, 'lon': lon} for lat, lon in result['path']],
                    'distance_km': round(result['distance_km'], 2),
                    'estimated_time_minutes': round(result['total_time_minutes'], 1),
                    'estimated_time_seconds': int(result['total_time_seconds']),
                    'segments': result['num_turns']
                }
            }
            print(json.dumps(output, indent=2))
        else:
            # Human-readable output
            logger.info("=" * 60)
            logger.info("📍 ROUTE FOUND")
            logger.info("=" * 60)
            logger.info(f"Source:      {source_lat:.4f}, {source_lon:.4f}")
            logger.info(f"Destination: {dest_lat:.4f}, {dest_lon:.4f}")
            logger.info("")
            logger.info(f"Distance:           {result['distance_km']:.2f} km")
            logger.info(f"Estimated time:     {result['total_time_minutes']:.1f} minutes ({result['total_time_seconds']:.0f}s)")
            logger.info(f"Number of segments: {result['num_turns']}")
            logger.info("")
            logger.info(f"Conditions:")
            logger.info(f"  Hour: {args.hour:02d}:00")
            logger.info(f"  Day: {['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][args.day]}")
            logger.info(f"  Raining: {'Yes' if args.raining else 'No'}")
            logger.info(f"  Festival: {'Yes' if args.festival else 'No'}")
            logger.info("")
            logger.info("First 5 waypoints:")
            for i, (lat, lon) in enumerate(result['path'][:5]):
                logger.info(f"  {i+1}. {lat:.6f}, {lon:.6f}")
            if len(result['path']) > 5:
                logger.info(f"  ... ({len(result['path']) - 5} more waypoints)")
            logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        logger.error("   Please run: python src/graph_loader.py")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Routing failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
