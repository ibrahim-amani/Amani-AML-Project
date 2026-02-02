import asyncio
import logging

# --- External Library Import ---
from adverse_media_parser.media_screening_engine import orchestrate_screening_cycle

class AdverseMediaService:
    def __init__(self):
        self.logger = logging.getLogger("Svc.AdverseMedia")

    def _sync_run(self):
        """
        Blocking function that runs the actual media engine.
        """
        self.logger.info("📰 [Adverse Media] Engine Started (CPU/IO Bound)...")
        try:
            orchestrate_screening_cycle(dry_run=False)
            self.logger.info("✅ [Adverse Media] Cycle Complete.")
            return "Media Success"
        except Exception as e:
            self.logger.error(f"❌ [Adverse Media] Failed: {e}")
            raise e

    async def run_async(self):
        """
        Offloads the blocking engine to a separate thread.
        """
        return await asyncio.to_thread(self._sync_run)