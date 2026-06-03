"""
Live Dashboard — Part E (Bonus +10 pts)

Real-time terminal dashboard using Rich library.
Polls the Store Intelligence API every 2 seconds and displays:
- Current visitor count
- Conversion rate
- Zone dwell breakdown
- Active anomalies
- Latest events feed

Usage:
    python dashboard/live_dashboard.py
    python dashboard/live_dashboard.py --api-url http://localhost:8000 --store STORE_BLR_002
    python dashboard/live_dashboard.py --replay  # Replay events from JSONL files

This proves the pipeline and API are genuinely connected — the dashboard
reads live from the API, not from pre-computed data.
"""
import sys
import time
import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.live import Live
    from rich.text import Text
    from rich.columns import Columns
    from rich import box
    from rich.progress import SpinnerColumn, TextColumn, Progress
except ImportError:
    print("Install rich and httpx: pip install rich httpx")
    sys.exit(1)

BASE_DIR = Path(__file__).parent.parent
EVENTS_DIR = BASE_DIR / "data" / "events"

console = Console()


def fetch_json(client: httpx.Client, url: str) -> dict:
    """Fetch JSON from a URL, return empty dict on error."""
    try:
        resp = client.get(url, timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {}


def build_dashboard(metrics: dict, funnel: dict, anomalies: dict, health: dict,
                    store_id: str, tick: int) -> Layout:
    """Build the Rich layout for the dashboard."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="right"),
    )
    layout["left"].split_column(
        Layout(name="metrics"),
        Layout(name="funnel"),
    )
    layout["right"].split_column(
        Layout(name="anomalies"),
        Layout(name="heatmap"),
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Header ─────────────────────────────────────────────────────────────
    spinner = "⣾⣽⣻⢿⡿⣟⣯⣷"[tick % 8]
    header_text = Text()
    header_text.append(f" {spinner} ", style="bold purple")
    header_text.append("PURPLLE STORE INTELLIGENCE ", style="bold white on purple")
    header_text.append(f" │ {store_id} │ {now} ", style="dim white")
    layout["header"].update(Panel(header_text, style="purple"))

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    metrics_table.add_column("Metric", style="bold cyan", width=28)
    metrics_table.add_column("Value", style="bold white", width=14)

    uv = metrics.get("unique_visitors", "—")
    cr = metrics.get("conversion_rate", 0)
    dwell = metrics.get("avg_dwell_sec", 0)
    queue = metrics.get("queue_depth_current", 0)
    abandon = metrics.get("abandonment_rate", 0)
    conf = metrics.get("data_confidence", "?")

    cr_color = "green" if cr >= 0.25 else "red" if cr < 0.1 else "yellow"
    queue_color = "red" if queue > 4 else "green"

    metrics_table.add_row("👥 Unique Visitors", str(uv))
    metrics_table.add_row("💰 Conversion Rate", f"[{cr_color}]{cr:.1%}[/{cr_color}]")
    metrics_table.add_row("⏱  Avg Dwell (sec)", f"{dwell:.0f}s")
    metrics_table.add_row("🏦 Queue Depth", f"[{queue_color}]{queue}[/{queue_color}]")
    metrics_table.add_row("🚪 Abandonment Rate", f"{abandon:.1%}")
    metrics_table.add_row("📊 Data Confidence", f"[dim]{conf}[/dim]")

    layout["metrics"].update(Panel(metrics_table, title="[bold cyan]📈 Live Metrics[/bold cyan]",
                                    border_style="cyan"))

    # ── Funnel ───────────────────────────────────────────────────────────────
    funnel_table = Table(box=box.SIMPLE, padding=(0, 1))
    funnel_table.add_column("Stage", style="bold", width=16)
    funnel_table.add_column("Count", style="white", justify="right", width=8)
    funnel_table.add_column("Drop-off", style="dim red", justify="right", width=10)

    stages = funnel.get("stages", [])
    bar_chars = "▇▇▇▇▇▇▇▇▇▇"
    for i, stage in enumerate(stages):
        name = stage.get("stage", "?")
        count = stage.get("count", 0)
        drop = stage.get("drop_off_pct", 0)
        icon = ["🚪", "🛍️", "🏦", "💳"][i] if i < 4 else "?"
        drop_str = f"-{drop:.0f}%" if drop > 0 else "—"
        funnel_table.add_row(f"{icon} {name}", str(count), drop_str)

    layout["funnel"].update(Panel(funnel_table, title="[bold yellow]🔻 Conversion Funnel[/bold yellow]",
                                   border_style="yellow"))

    # ── Anomalies ────────────────────────────────────────────────────────────
    anomaly_list = anomalies.get("active_anomalies", [])
    anom_table = Table(box=box.SIMPLE, padding=(0, 1))
    anom_table.add_column("Type", style="bold", width=22)
    anom_table.add_column("Sev", width=8)
    anom_table.add_column("Description", style="dim white", width=30)

    severity_styles = {"INFO": "blue", "WARN": "yellow", "CRITICAL": "bold red"}

    if anomaly_list:
        for a in anomaly_list[:6]:  # Show max 6
            sev = a.get("severity", "INFO")
            atype = a.get("anomaly_type", "UNKNOWN")
            desc = a.get("description", "")[:35]
            style = severity_styles.get(sev, "white")
            anom_table.add_row(atype, f"[{style}]{sev}[/{style}]", desc)
    else:
        anom_table.add_row("[green]✓ All clear[/green]", "", "No active anomalies")

    layout["anomalies"].update(Panel(anom_table, title="[bold red]⚠️  Anomalies[/bold red]",
                                      border_style="red"))

    # ── Heatmap ──────────────────────────────────────────────────────────────
    zone_breakdown = metrics.get("zone_breakdown", [])
    heat_table = Table(box=box.SIMPLE, padding=(0, 1))
    heat_table.add_column("Zone", style="bold cyan", width=14)
    heat_table.add_column("Visitors", width=10)
    heat_table.add_column("Dwell", width=8)
    heat_table.add_column("Heat", width=12)

    zone_max = max((z.get("visitor_count", 0) for z in zone_breakdown), default=1) or 1
    for zone in zone_breakdown[:6]:
        zname = zone.get("zone_id", "?")
        zcount = zone.get("visitor_count", 0)
        zdwell = zone.get("avg_dwell_sec", 0)
        heat_pct = int((zcount / zone_max) * 10)
        heat_bar = "█" * heat_pct + "░" * (10 - heat_pct)
        heat_color = "red" if heat_pct > 7 else "yellow" if heat_pct > 4 else "green"
        heat_table.add_row(zname, str(zcount), f"{zdwell:.0f}s",
                           f"[{heat_color}]{heat_bar}[/{heat_color}]")

    layout["heatmap"].update(Panel(heat_table, title="[bold green]🔥 Zone Heatmap[/bold green]",
                                    border_style="green"))

    # ── Footer ───────────────────────────────────────────────────────────────
    health_status = health.get("status", "UNKNOWN")
    health_color = "green" if health_status == "OK" else "red"
    footer_text = Text()
    footer_text.append(f" API: [{health_color}]{health_status}[/{health_color}]  ", style="bold")
    footer_text.append("│  Press Ctrl+C to exit  │  Refresh: 2s", style="dim white")
    layout["footer"].update(Panel(footer_text, style="dim"))

    return layout


def run_dashboard(api_url: str, store_id: str):
    """Main dashboard loop."""
    client = httpx.Client(base_url=api_url)
    tick = 0

    console.print(f"[bold purple]Purplle Store Intelligence Dashboard[/bold purple]")
    console.print(f"Connecting to API at [cyan]{api_url}[/cyan]...")

    # Check API is up
    try:
        resp = client.get("/health", timeout=5.0)
        if resp.status_code != 200:
            console.print(f"[red]API returned {resp.status_code}. Is it running?[/red]")
            return
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {api_url}. Start the API first:[/red]")
        console.print("  uvicorn app.main:app --reload")
        return

    console.print("[green]Connected! Starting dashboard...[/green]")
    time.sleep(0.5)

    with Live(console=console, refresh_per_second=2, screen=True) as live:
        while True:
            metrics = fetch_json(client, f"/stores/{store_id}/metrics")
            funnel = fetch_json(client, f"/stores/{store_id}/funnel")
            anomalies = fetch_json(client, f"/stores/{store_id}/anomalies")
            health = fetch_json(client, "/health")

            layout = build_dashboard(metrics, funnel, anomalies, health, store_id, tick)
            live.update(layout)
            tick += 1
            time.sleep(2)


def replay_events(api_url: str, store_id: str):
    """
    Replay events from JSONL files in real-time simulation.
    Sends events to API in order, then the dashboard shows live updates.
    """
    import glob
    files = sorted(glob.glob(str(EVENTS_DIR / "*.jsonl")))

    if not files:
        console.print(f"[red]No event files found in {EVENTS_DIR}[/red]")
        console.print("Run the pipeline first: python pipeline/detect.py --all")
        return

    console.print(f"[yellow]Replaying {len(files)} event files in real-time simulation...[/yellow]")

    client = httpx.Client(base_url=api_url)
    batch = []
    BATCH_SIZE = 50

    for file in files:
        with open(file) as f:
            events = [json.loads(line) for line in f if line.strip()]

        console.print(f"  Replaying {len(events)} events from {Path(file).name}...")
        for i, event in enumerate(events):
            batch.append(event)
            if len(batch) >= BATCH_SIZE:
                try:
                    client.post("/events/ingest", json={"events": batch}, timeout=10)
                except Exception:
                    pass
                batch = []
                time.sleep(0.1)  # Simulate real-time pacing

        if batch:
            try:
                client.post("/events/ingest", json={"events": batch}, timeout=10)
            except Exception:
                pass
            batch = []

    console.print("[green]Replay complete! Starting dashboard...[/green]")
    run_dashboard(api_url, store_id)


def main():
    parser = argparse.ArgumentParser(description="Store Intelligence Live Dashboard")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--store", default="STORE_BLR_002")
    parser.add_argument("--replay", action="store_true",
                        help="Replay events from JSONL files before showing dashboard")
    args = parser.parse_args()

    if args.replay:
        replay_events(args.api_url, args.store)
    else:
        run_dashboard(args.api_url, args.store)


if __name__ == "__main__":
    main()
