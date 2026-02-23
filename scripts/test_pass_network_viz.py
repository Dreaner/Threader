#!/usr/bin/env python3
"""
Quick visual test for pass network visualization.

Loads one World Cup match, builds a pass network for both teams,
computes metrics, and saves one interactive HTML figure per team.
The figure includes Plain / Degree / Betweenness / PageRank toggle buttons.

Usage:
    python scripts/test_pass_network_viz.py
    python scripts/test_pass_network_viz.py --game 10502 --min-edge 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pitch_echo.data.pff.events import extract_pass_events
from pitch_echo.data.pff.metadata import load_match_info
from pitch_echo.network import build_pass_network, compute_metrics
from pitch_echo.viz.plotly_network import build_network_figure

DATA_ROOT = _project_root / "data" / "FIFA_World_Cup_2022"
EVENT_DIR = DATA_ROOT / "Event Data"
META_DIR = DATA_ROOT / "Metadata"
OUT_DIR = _project_root / "output" / "pass_network"

# JavaScript injected after the figure renders.
# Makes the Plotly div (and its page) fill the entire browser viewport.
_VIEWPORT_JS = (
    "var s=document.createElement('style');"
    "s.textContent="
    "'*{box-sizing:border-box}"
    "html,body{margin:0;padding:0;height:100%;overflow:hidden;background:#0f0f1a}"
    ".plotly-graph-div{height:100vh!important;width:100vw!important}';"
    "document.head.appendChild(s);"
)


def _team_label(match_info, team_id: int) -> str:
    if match_info.home_team.team_id == team_id:
        return match_info.home_team.name
    return match_info.away_team.name


def run(game_id: int, min_edge: int) -> None:
    event_path = EVENT_DIR / f"{game_id}.json"
    meta_path = META_DIR / f"{game_id}.json"

    if not event_path.exists():
        print(f"ERROR: event file not found: {event_path}")
        sys.exit(1)

    print(f"Loading game {game_id}...")
    match_info = load_match_info(meta_path)
    events = extract_pass_events(event_path, meta_path)

    home_id = int(match_info.home_team.team_id)
    away_id = int(match_info.away_team.team_id)
    home_name = match_info.home_team.name
    away_name = match_info.away_team.name

    print(f"  {home_name} ({home_id}) vs {away_name} ({away_id})")
    print(f"  {len(events)} pass events")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for team_id, team_name in [(home_id, home_name), (away_id, away_name)]:
        print(f"\n--- {team_name} ---")

        network = build_pass_network(events, team_id, completed_only=True)
        metrics = compute_metrics(network)

        print(f"  nodes: {len(network.nodes)}, edges: {len(network.edges)}")
        print(f"  density: {metrics.density:.3f}")
        print(f"  total completed passes: {network.total_passes}")

        print("  Top hubs (degree):")
        for pm in metrics.top_hubs(3):
            node = network.nodes.get(pm.player_id)
            label = f"#{node.jersey_num} {node.name or pm.player_id}" if node else pm.player_id
            print(f"    {label}: degree={pm.degree_centrality:.3f}")

        print("  Top connectors (betweenness):")
        for pm in metrics.top_connectors(3):
            node = network.nodes.get(pm.player_id)
            label = f"#{node.jersey_num} {node.name or pm.player_id}" if node else pm.player_id
            print(f"    {label}: betweenness={pm.betweenness_centrality:.3f}")

        print("  Top receivers (pagerank):")
        for pm in metrics.top_receivers(3):
            node = network.nodes.get(pm.player_id)
            label = f"#{node.jersey_num} {node.name or pm.player_id}" if node else pm.player_id
            print(f"    {label}: pagerank={pm.pagerank:.3f}")

        slug = team_name.lower().replace(" ", "_")
        fig = build_network_figure(
            network,
            metrics,
            min_edge_count=min_edge,
            title=f"{team_name} — Pass Network",
        )
        out_path = OUT_DIR / f"{game_id}_{slug}.html"
        fig.write_html(
            str(out_path),
            include_plotlyjs="cdn",
            config={"responsive": True},
            post_script=_VIEWPORT_JS,
        )
        print(f"  Saved: {out_path.relative_to(_project_root)}")

    print(f"\nDone. Open files in output/pass_network/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pass network visual test")
    parser.add_argument("--game", type=int, default=10502,
                        help="Game ID (default: 10502 = Netherlands vs USA)")
    parser.add_argument("--min-edge", type=int, default=2,
                        help="Min edge count to show (default: 2)")
    args = parser.parse_args()
    run(args.game, args.min_edge)


if __name__ == "__main__":
    main()
