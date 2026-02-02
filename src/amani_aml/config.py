import logging
from dataclasses import dataclass
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]

@dataclass
class AppConfig:
    DATA_PATH: str = str(PROJECT_ROOT / "data")
    SANCTIONS_CONCURRENCY: int = 3  # Max parallel scrapers
    LOG_LEVEL: str = "INFO"

# Global settings instance
settings = AppConfig()

def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )