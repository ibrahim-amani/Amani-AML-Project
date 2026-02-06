import logging
from pathlib import Path
from pydantic_settings import BaseSettings

# Define Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Project Info
    # ------------------------------------------------------------------
    PROJECT_NAME: str = "AML Project"
    PROVIDER_NAME:str="AmaniAI"
    
    CONTRACT_VERSION:str="1.0"
    SCEMA_VERSION:str="1.0"
    # ------------------------------------------------------------------
    # App Configuration (Moved from AppConfig)
    # ------------------------------------------------------------------
    SANCTIONS_CONCURRENCY: int = 3  # Max parallel scrapers
    LOG_LEVEL: str = "INFO"

    # ------------------------------------------------------------------
    # Data Lake Configuration
    # ------------------------------------------------------------------
    # Default to current working directory / data
    DATA_LAKE_DIR: Path = Path.cwd() / "data"

    @property
    def RAW_DIR(self) -> Path:
        return self.DATA_LAKE_DIR / "raw"

    @property
    def PROCESSED_DIR(self) -> Path:
        return self.DATA_LAKE_DIR / "schema"
    
    # ------------------------------------------------------------------
    # Relationship Caps (Graph Safety)
    # ------------------------------------------------------------------
    MAX_RELS_PER_ENTITY: int = 300
    MAX_REL_ENDPOINTS_PER_SIDE: int = 50
    
    # ------------------------------------------------------------------
    # Pydantic Config
    # ------------------------------------------------------------------
    class Config:
        env_file = ".env"
        env_prefix = "SANCTION_"
        case_sensitive = True
        validate_assignment = True

# ----------------------------------------------------------------------
# Singleton Settings Instance
# ----------------------------------------------------------------------
settings = Settings()

def setup_logging():
    """Configures the root logger for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
def set_data_lake_path(new_path: str):
    """
    Updates the Data Lake path at runtime (CLI / API usage).
    """
    path = Path(new_path).resolve()
    settings.DATA_LAKE_DIR = path

    # Ensure directories exist immediately
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)