import asyncio
import logging
from typing import List

# --- External Library Import ---
from sanction_parser.scrapers.registry import ScraperRegistry

class SanctionsScraperService:
    def __init__(self, concurrency_limit: int):
        self.limit = concurrency_limit
        self.logger = logging.getLogger("Svc.SanctionsScraper")

    async def _run_single_scraper(self, key: str, sem: asyncio.Semaphore):
        """Runs a single scraper protected by a semaphore."""
        async with sem:
            try:
                self.logger.info(f"📡 [Sanctions] Scraping source: {key}")
                scraper = ScraperRegistry.get_scraper(key)
                await scraper.run(force=False)
                return f"Success: {key}"
            except Exception as e:
                self.logger.error(f"❌ [Sanctions] Source failed [{key}]: {e}")
                return e  # Return exception instead of raising to keep others running

    async def run_scrapers(self) -> List:
        """
        Orchestrates the scraping of all available sources concurrently.
        """
        keys = ScraperRegistry.list_keys()
        self.logger.info(f"🚀 [Sanctions] Starting update for {len(keys)} sources...")
        
        sem = asyncio.Semaphore(self.limit)
        
        # Create tasks
        tasks = [self._run_single_scraper(key, sem) for key in keys]
        
        # Run all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        self.logger.info("✅ [Sanctions] Scraping phase finished.")
        return results