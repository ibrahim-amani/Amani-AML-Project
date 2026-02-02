import asyncio
import logging

# --- External Library Import ---
from sanction_parser.pipelines.resolver import EntityResolver

class ResolverService:
    def __init__(self):
        self.logger = logging.getLogger("Svc.Resolver")

    def _sync_run(self):
        """
        Blocking function that runs the entity resolution logic.
        """
        self.logger.info("🔗 [Resolver] Starting Entity Resolution Phase...")
        try:
            resolver = EntityResolver()
            resolver.run()
            self.logger.info("✨ [Resolver] Golden Records generated.")
            return "Resolver Success"
        except Exception as e:
            self.logger.error(f"❌ [Resolver] Failed: {e}")
            raise e

    async def run_async(self):
        """
        Offloads the blocking resolver to a separate thread.
        """
        return await asyncio.to_thread(self._sync_run)