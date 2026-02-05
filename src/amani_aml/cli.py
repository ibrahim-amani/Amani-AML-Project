from __future__ import annotations

import sys
import platform
import asyncio
import time
import logging
from typing import Optional, Any, List, TYPE_CHECKING
from importlib.metadata import version as pkg_version, PackageNotFoundError

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from rich import box

# --- TYPE HINTS (No Runtime Cost) ---
if TYPE_CHECKING:
    from .services.sanctions_svc import SanctionsScraperService
    from .services.media_svc import AdverseMediaService
    from .services.resolver_svc import ResolverService

# --- CONFIG (Lightweight) ---
# We assume settings is light. If config.py is heavy, move this inside functions too.
from .core.config import settings

# --- UI INIT ---
console = Console()
app = typer.Typer(
    help="🛡️ Amani AML Project CLI (Sanctions + Adverse Media + Resolver)",
    rich_markup_mode="rich",
)
logger = logging.getLogger("amani-aml")


# ============================================================
# UI Helpers
# ============================================================
def print_banner():
    """Prints the high-impact AMANI AML ASCII system header."""
    try:
        app_version = pkg_version("Amani-AML-Project")
    except PackageNotFoundError:
        app_version = "dev"

    table = Table(show_header=False, box=None, padding=(0, 1))
    ascii_art = [
        r"[bold cyan]     _    __  __    _    _   _ ___        _    __  __ _     [/bold cyan]",
        r"[bold cyan]    / \  |  \/  |  / \  | \ | |_ _|      / \  |  \/  | |    [/bold cyan]",
        r"[bold cyan]   / _ \ | |\/| | / _ \ |  \| || |      / _ \ | |\/| | |    [/bold cyan]",
        r"[bold cyan]  / ___ \| |  | |/ ___ \| |\  || |     / ___ \| |  | | |___ [/bold cyan]",
        r"[bold cyan] /_/   \_\_|  |_/_/   \_\_| \_|___|   /_/   \_\_|  |_|_____|[/bold cyan]",
        r"[bold white]        A N T I   M O N E Y   L A U N D E R I N G               [/bold white]",
    ]
    for line in ascii_art:
        table.add_row(line)
    
    table.add_row("")
    table.add_row(f"[bold yellow]        AMANI AML Platform • Intelligence & Risk Screening v{app_version}[/bold yellow]")
    console.print("")
    console.print(table)
    console.print("[dim] -----------------------------------------------------------------------[/dim]")

def _phase_panel(title: str, body: str) -> None:
    console.print(Panel(body, title=title, border_style="magenta"))

def _summary_table(rows: List[List[str]]) -> None:
    t = Table(title="Run Summary", box=box.SIMPLE_HEAVY, header_style="bold magenta")
    t.add_column("Component", style="cyan", no_wrap=True)
    t.add_column("Status", style="white")
    t.add_column("Details", style="dim")
    for r in rows:
        t.add_row(*r)
    console.print(t)

def _as_status(ok: bool) -> str:
    return "[green]OK[/green]" if ok else "[red]FAIL[/red]"


# ============================================================
# Orchestration (Async + Lazy Loading)
# ============================================================
async def orchestrate_pipeline(
    run_sanctions: bool = True,
    run_media: bool = True,
    run_resolver: bool = True,
    parallel: Optional[int] = None,
) -> None:
    """
    Phase 1 (parallel): sanctions scrapers + adverse media
    Phase 2 (sequential): resolver
    """
    # 💤 LAZY IMPORTS: Load heavy libraries only when pipeline starts
    from amani_aml.core.config import setup_logging
    from amani_aml.services.sanctions_svc import SanctionsScraperService
    from amani_aml.services.media_svc import AdverseMediaService
    from amani_aml.services.resolver_svc import ResolverService
    from amani_aml.core.config import set_data_lake_path, settings as lib_settings

    # Initialize Logging only now
    setup_logging()
    
    # Configure Rich Logging specifically for this run
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
        force=True # Force override of any existing logging config
    )
    orch_logger = logging.getLogger("Orchestrator")

    # Ensure sanction_parser library uses your configured data path
    set_data_lake_path(settings.DATA_LAKE_DIR)

    start_ts = time.time()
    orch_logger.info("=== STARTING JOINT AML OPERATIONS ===")
    orch_logger.info(f"DATA_LAKE_DIR: {lib_settings.DATA_LAKE_DIR}")

    # Init services
    sanctions_svc = SanctionsScraperService(concurrency_limit=parallel or settings.SANCTIONS_CONCURRENCY)
    media_svc = AdverseMediaService()
    resolver_svc = ResolverService()

    summary_rows: List[List[str]] = []
    
    # ----------------------------
    # Phase 1: Parallel Fetching
    # ----------------------------
    orch_logger.info("--- PHASE 1: Fetching Data (Sanctions & Media) ---")

    tasks = []
    task_names = []

    if run_sanctions:
        tasks.append(asyncio.create_task(sanctions_svc.run_scrapers()))
        task_names.append("sanctions")

    if run_media:
        tasks.append(asyncio.create_task(media_svc.run_async()))
        task_names.append("media")

    results: List[Any] = []
    if tasks:
        # Run both engines simultaneously
        results = await asyncio.gather(*tasks, return_exceptions=True)

    sanctions_failed = False

    # Interpret results
    for name, res in zip(task_names, results):
        if name == "sanctions":
            if isinstance(res, Exception):
                sanctions_failed = True
                summary_rows.append(["Sanctions/PEP", _as_status(False), f"{res}"])
                orch_logger.error(f"❌ Sanctions phase crashed: {res}")
            else:
                # Count individual scraper successes/failures
                ok = sum(1 for item in res if not isinstance(item, Exception))
                failed = len(res) - ok
                summary_rows.append(["Sanctions/PEP", _as_status(failed == 0), f"ok={ok}, failed={failed}"])
                
        elif name == "media":
            if isinstance(res, Exception):
                summary_rows.append(["Adverse Media", _as_status(False), f"{res}"])
                orch_logger.error(f"⚠️ Adverse Media failed: {res}")
            else:
                summary_rows.append(["Adverse Media", _as_status(True), "Completed"])

    # ----------------------------
    # Phase 2: Resolver
    # ----------------------------
    if run_resolver:
        orch_logger.info("--- PHASE 2: Entity Resolution ---")

        if run_sanctions and sanctions_failed:
            orch_logger.error("⛔ Skipping Resolver because Sanctions scrapers crashed.")
            summary_rows.append(["Resolver", _as_status(False), "skipped (sanctions failed)"])
        else:
            try:
                res = await resolver_svc.run_async()
                summary_rows.append(["Resolver", _as_status(True), "Golden Records Generated"])
            except Exception as e:
                summary_rows.append(["Resolver", _as_status(False), str(e)])
                orch_logger.error(f"❌ Resolver crashed: {e}")

    # ----------------------------
    # Summary
    # ----------------------------
    duration = time.time() - start_ts
    orch_logger.info(f"=== ALL PIPELINES FINISHED IN {duration:.2f} SECONDS ===")
    orch_logger.info(f"Check results in: {lib_settings.DATA_LAKE_DIR}")

    _summary_table(summary_rows)
    console.print(Panel(f"[bold green]Total time:[/] {duration:.2f}s\n[bold]Data Lake:[/] {lib_settings.DATA_LAKE_DIR}", border_style="green"))


# ============================================================
# CLI Commands
# ============================================================
@app.command("version")
def version_cmd():
    """Show Amani AML CLI version information."""
    print_banner()
    try:
        app_version = pkg_version("Amani-AML-Project")
    except PackageNotFoundError:
        app_version = "unknown (not installed)"

    console.print(
        Panel(
            f"[bold cyan]Amani AML Project[/]\n"
            f"[bold]Version:[/] {app_version}\n"
            f"[bold]Python:[/] {sys.version.split()[0]}\n"
            f"[bold]Platform:[/] {platform.system()} {platform.release()}",
            title="Version Info",
            border_style="green",
        )
    )


@app.command("sources")
def list_all_sources(
    sanctions: bool = typer.Option(False, "--sanctions", help="Show sanctions / PEP sources only"),
    media: bool = typer.Option(False, "--media", help="Show adverse media sources only"),
    all: bool = typer.Option(False, "--all", help="Show all sources (default)"),
):
    """List all available AML data sources (Sanctions/PEP + Adverse Media)."""
    # 💤 LAZY IMPORT
    from adverse_media_parser.core.source_registry import SourceRegistry as MediaSourceRegistry
    from sanction_parser.scrapers.registry import ScraperRegistry

    print_banner()

    show_sanctions = sanctions or all or (not sanctions and not media and not all)
    show_media = media or all or (not sanctions and not media and not all)

    # Sanctions Table
    if show_sanctions:
        keys = ScraperRegistry.list_keys()
        t = Table(title="Sanctions & PEP Sources", box=box.SIMPLE_HEAVY, header_style="bold magenta")
        t.add_column("Key", style="cyan", no_wrap=True)
        t.add_column("Provider Name", style="white")
        t.add_column("Status", justify="center")

        for key in keys:
            try:
                s = ScraperRegistry.get_scraper(key)
                name = getattr(s, "name", "Unknown")
                status = "[green]● ACTIVE[/green]"
            except Exception:
                name = "N/A"
                status = "[red]● ERROR[/red]"
            t.add_row(key, name, status)

        console.print(t)
        console.print(Panel(f"Total Sanctions/PEP Sources: [bold cyan]{len(keys)}[/]", border_style="blue"))

    # Media Table
    if show_media:
        media_sources = MediaSourceRegistry.get_all()
        table = Table(title="Adverse Media Sources", box=box.SIMPLE_HEAVY, header_style="bold magenta")
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Region", style="magenta")
        table.add_column("Provider / Source Name", style="white")
        table.add_column("Status", justify="center")

        for key, data in media_sources.items():
            table.add_row(
                key,
                data.get("region", "-"),
                f"{data.get('provider', '-') } - {data.get('name', '-')}",
                data.get("type", "-"),
                "[green]● Online[/green]",
            )

        console.print(table)
        console.print(Panel(f"Total Adverse Media Sources: [bold cyan]{len(media_sources)}[/]", border_style="blue"))


@app.command("pipeline")
def pipeline_run(
    parallel: int = typer.Option(None, "--parallel", "-p", help="Max concurrent sanctions scrapers"),
    no_sanctions: bool = typer.Option(False, "--no-sanctions", help="Disable sanctions/PEP scraping"),
    no_media: bool = typer.Option(False, "--no-media", help="Disable adverse media screening"),
    no_resolve: bool = typer.Option(False, "--no-resolve", help="Disable entity resolution"),
):
    """Run full pipeline: Phase 1 (parallel) + Phase 2 (resolver)."""
    print_banner()
    _phase_panel(
        "Pipeline Plan",
        f"[bold]Sanctions:[/] {'OFF' if no_sanctions else 'ON'}\n"
        f"[bold]Media:[/] {'OFF' if no_media else 'ON'}\n"
        f"[bold]Resolve:[/] {'OFF' if no_resolve else 'ON'}\n"
        f"[bold]Parallel:[/] {parallel or settings.SANCTIONS_CONCURRENCY}\n"
        f"[bold]DATA_LAKE_DIR:[/] {settings.DATA_LAKE_DIR}",
    )
    asyncio.run(
        orchestrate_pipeline(
            run_sanctions=not no_sanctions,
            run_media=not no_media,
            run_resolver=not no_resolve,
            parallel=parallel,
        )
    )

@app.command("sanctions")
def sanctions_only(
    parallel: int = typer.Option(None, "--parallel", "-p", help="Max concurrent scrapers"),
):
    """Run only sanctions/PEP scraping."""
    print_banner()
    asyncio.run(orchestrate_pipeline(run_sanctions=True, run_media=False, run_resolver=False, parallel=parallel))

@app.command("media")
def media_only():
    """Run only adverse media screening."""
    print_banner()
    asyncio.run(orchestrate_pipeline(run_sanctions=False, run_media=True, run_resolver=False))

@app.command("resolve")
def resolve_only():
    """Run only entity resolution (assumes data already collected)."""
    print_banner()
    asyncio.run(orchestrate_pipeline(run_sanctions=False, run_media=False, run_resolver=True))

@app.command("api")
def api_run(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload (dev only)"),
):
    """Start the FastAPI export service."""
    # 💤 LAZY IMPORT
    import uvicorn
    from .core.config import set_data_lake_path, settings as lib_settings
    
    print_banner()
    # keep the data path consistent
    set_data_lake_path(settings.DATA_LAKE_DIR)

    console.print(
        Panel(
            f"[bold]Starting API[/]\n"
            f"Host: [cyan]{host}[/]  Port: [cyan]{port}[/]\n"
            f"Data Lake: [dim]{lib_settings.DATA_LAKE_DIR}[/]",
            border_style="green",
        )
    )

    uvicorn.run(
        "amani_aml.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


# ============================================================
# Entry point
# ============================================================
def main() -> None:
    app()

if __name__ == "__main__":
    main()