"""
Project: Threader
File Created: 2026-02-16
Author: Xingnan Zhu
File Name: app.py
Description:
    Dash web application for interactive pass analysis.
    Left panel: Plotly pitch with players and pass arrows.
    Right panel: Analysis results with ranked pass options.

    Usage:
        uv run threader              # via entry point
        uv run python -m threader.app  # direct
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, dcc, html, no_update

from threader.data.pff.events import PassEvent, extract_pass_events
from threader.data.pff.metadata import load_match_info
from threader.data.pff.tracking_frames import get_animation_frames_cached
from threader.metrics.pass_value.analyzer import analyze_pass_event
from threader.metrics.pass_value.models import AnalysisResult
from threader.viz.plotly_animation_3d import build_animation_figure_3d
from threader.viz.plotly_passes import (
    build_analysis_figure as build_static_figure,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_ROOT = Path("data/FIFA_World_Cup_2022")
ASSETS_DIR = Path(__file__).parent / "assets"

# Rank colors matching the pitch arrows
RANK_COLORS = ["#FFD700", "#C0C0C0", "#CD7F32", "#888888", "#666666"]

# Score color thresholds
def _score_color(score: float) -> str:
    if score >= 30:
        return "#2ecc71"
    elif score >= 20:
        return "#f1c40f"
    else:
        return "#e74c3c"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _discover_matches() -> list[dict]:
    """Scan the metadata directory for available matches."""
    meta_dir = DATA_ROOT / "Metadata"
    event_dir = DATA_ROOT / "Event Data"
    matches = []

    if not meta_dir.exists():
        return matches

    for meta_file in sorted(meta_dir.glob("*.json")):
        game_id = meta_file.stem
        event_file = event_dir / f"{game_id}.json"
        if not event_file.exists():
            continue
        try:
            info = load_match_info(str(meta_file))
            label = f"{info.home_team.name} vs {info.away_team.name}"
            matches.append({"label": label, "value": game_id, "info": info})
        except Exception:
            continue

    return matches


@lru_cache(maxsize=8)
def _load_passes(game_id: str) -> list[PassEvent]:
    """Load and cache pass events for a match."""
    return extract_pass_events(
        str(DATA_ROOT / "Event Data" / f"{game_id}.json"),
        str(DATA_ROOT / "Metadata" / f"{game_id}.json"),
    )


@lru_cache(maxsize=8)
def _get_match_info(game_id: str):
    return load_match_info(str(DATA_ROOT / "Metadata" / f"{game_id}.json"))


# Pre-discover matches at import time
MATCHES = _discover_matches()


# ---------------------------------------------------------------------------
# Dash App
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    assets_folder=str(ASSETS_DIR),
    external_stylesheets=[dbc.themes.DARKLY],
    title="Threader — Pass Analysis",
    update_title=None,
    suppress_callback_exceptions=True,
)

# Expose WSGI server for deployment (gunicorn threader.app:server)
server = app.server


# ---------------------------------------------------------------------------
# Layout Helpers
# ---------------------------------------------------------------------------
def _build_header() -> html.Div:
    return html.Div(
        className="app-header",
        children=[
            html.Div([
                html.H1("THREADER", className="app-title"),
                html.P("AlphaGo-inspired Pass Analysis", className="app-subtitle"),
            ]),
            html.Div(
                "FIFA World Cup 2022",
                style={
                    "color": "#9999b3",
                    "fontSize": "0.8rem",
                    "fontWeight": "600",
                },
            ),
        ],
    )


def _build_selectors() -> html.Div:
    match_options = [
        {"label": m["label"], "value": m["value"]} for m in MATCHES
    ]
    default_match = MATCHES[0]["value"] if MATCHES else None

    return html.Div(
        className="selector-row",
        children=[
            dbc.Row([
                dbc.Col([
                    html.Div("Match", className="selector-label"),
                    dcc.Dropdown(
                        id="match-selector",
                        options=match_options,
                        value=default_match,
                        clearable=False,
                        style={"backgroundColor": "#1c1c35", "color": "#e8e8f0"},
                    ),
                ], md=5),
                dbc.Col([
                    html.Div("Pass Event", className="selector-label"),
                    dcc.Dropdown(
                        id="pass-selector",
                        clearable=False,
                        style={"backgroundColor": "#1c1c35", "color": "#e8e8f0"},
                    ),
                ], md=5),
                dbc.Col([
                    html.Div("Show", className="selector-label"),
                    dcc.Dropdown(
                        id="top-n-selector",
                        options=[
                            {"label": "Top 3", "value": 3},
                            {"label": "Top 5", "value": 5},
                            {"label": "All", "value": 99},
                        ],
                        value=3,
                        clearable=False,
                        style={"backgroundColor": "#1c1c35", "color": "#e8e8f0"},
                    ),
                ], md=2),
            ]),
        ],
    )


def _build_main() -> html.Div:
    return html.Div(
        className="main-content",
        children=[
            # Left: Pitch
            html.Div(
                className="pitch-column",
                children=[
                    dcc.Graph(
                        id="pitch-graph",
                        className="pitch-graph",
                        style={"height": "calc(100vh - 140px)"},
                        config={
                            "displayModeBar": False,
                            "staticPlot": False,
                            "scrollZoom": False,
                        },
                    ),
                ],
            ),
            # Right: Analysis panel
            html.Div(
                className="analysis-panel",
                children=[
                    html.Button(
                        id="play-animation-btn",
                        className="play-animation-btn",
                        n_clicks=0,
                        style={"display": "none"},
                        children=[
                            html.Span("\u25b6", className="play-icon"),
                            html.Span("Play Pass", className="play-label"),
                        ],
                    ),
                    html.Div(
                        id="analysis-panel",
                        children=[
                            html.Div(
                                "Select a match and pass event to begin",
                                className="loading-placeholder",
                            )
                        ],
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------
app.layout = html.Div(
    className="app-container",
    children=[
        _build_header(),
        _build_selectors(),
        _build_main(),
        # Hidden stores
        dcc.Store(id="selected-option-idx", data=None),
        dcc.Store(id="analysis-cache", data=None),
        dcc.Store(id="animation-loading", data=False),
        dcc.Store(id="autoplay-signal", data=0),
        # dummy output for autoplay clientside callback
        dcc.Store(id="autoplay-done", data=0),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# Auto-play animation when the figure loads with frames
app.clientside_callback(
    """
    function(signal) {
        if (!signal) return window.dash_clientside.no_update;
        // Wait for Plotly to finish rendering the new figure
        setTimeout(function() {
            var gd = document.getElementById('pitch-graph');
            if (gd && gd._fullLayout && gd._transitionData &&
                gd._transitionData._frames &&
                gd._transitionData._frames.length > 0) {
                Plotly.animate('pitch-graph', null, {
                    frame: {duration: 100, redraw: true},
                    transition: {duration: 80},
                    fromcurrent: true,
                    mode: 'immediate'
                });
            }
        }, 300);
        return window.dash_clientside.no_update;
    }
    """,
    Output("autoplay-done", "data"),
    Input("autoplay-signal", "data"),
    prevent_initial_call=True,
)

# Instant loading feedback when Play Pass is clicked (before server round-trip)
# Returns disabled + className through Dash so Dash tracks the state properly.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) return [window.dash_clientside.no_update,
                              window.dash_clientside.no_update,
                              window.dash_clientside.no_update];
        return [true, "play-animation-btn loading", "loading"];
    }
    """,
    Output("play-animation-btn", "disabled"),
    Output("play-animation-btn", "className"),
    Output("animation-loading", "data"),
    Input("play-animation-btn", "n_clicks"),
    prevent_initial_call=True,
)

@app.callback(
    Output("pass-selector", "options"),
    Output("pass-selector", "value"),
    Input("match-selector", "value"),
)
def update_pass_list(game_id: str):
    """When match changes, load pass events and populate the dropdown."""
    if not game_id:
        return [], None

    passes = _load_passes(game_id)
    options = []
    for i, pe in enumerate(passes):
        clock_min = pe.game_clock // 60
        clock_sec = pe.game_clock % 60
        outcome = "\u2713" if pe.is_complete else "\u2717"
        label = (
            f"P{pe.period} {clock_min:02d}:{clock_sec:02d}  "
            f"{pe.passer_name} \u2192 {pe.target_name}  {outcome}"
        )
        options.append({"label": label, "value": i})

    default = 5 if len(options) > 5 else 0
    return options, default


@app.callback(
    Output("pitch-graph", "figure"),
    Output("analysis-panel", "children"),
    Output("analysis-cache", "data"),
    Output("play-animation-btn", "style"),
    Output("play-animation-btn", "disabled", allow_duplicate=True),
    Output("play-animation-btn", "className", allow_duplicate=True),
    Input("pass-selector", "value"),
    Input("top-n-selector", "value"),
    Input("selected-option-idx", "data"),
    State("match-selector", "value"),
    prevent_initial_call="initial_duplicate",
)
def update_analysis(pass_idx, top_n, selected_idx, game_id):
    """Main callback: render pitch and analysis panel."""
    if game_id is None or pass_idx is None:
        empty_fig = {
            "data": [],
            "layout": {
                "plot_bgcolor": "#1a472a",
                "paper_bgcolor": "#0f0f1a",
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
            },
        }
        placeholder = html.Div(
            "Select a match and pass event to begin",
            className="loading-placeholder",
        )
        return (
            empty_fig, placeholder, None,
            {"display": "none"}, False, "play-animation-btn",
        )

    # Load data
    passes = _load_passes(game_id)
    pass_event = passes[pass_idx]
    match_info = _get_match_info(game_id)

    # Run analysis
    result = analyze_pass_event(pass_event)

    # Build pitch figure
    home = match_info.home_team.short_name
    away = match_info.away_team.short_name
    title = f"{home} vs {away} \u2014 {pass_event.passer_name}"
    fig = build_static_figure(
        result,
        top_n=top_n,
        show_all=True,
        title=title,
        selected_idx=selected_idx,
    )
    fig.update_layout(
        height=None,
        autosize=True,
    )

    # Build analysis panel
    panel = _build_analysis_panel(result, pass_event, match_info, selected_idx)

    # Cache analysis data for click interaction
    cache = {
        "game_id": game_id,
        "pass_idx": pass_idx,
        "n_options": len(result.ranked_options),
    }

    return fig, panel, cache, {"display": "block"}, False, "play-animation-btn"


@app.callback(
    Output("selected-option-idx", "data"),
    Input({"type": "option-card", "index": dash.ALL}, "n_clicks"),
    State("selected-option-idx", "data"),
    prevent_initial_call=True,
)
def handle_card_click(n_clicks_list, current_idx):
    """Handle clicks on pass option cards."""
    if not ctx.triggered_id:
        return no_update
    clicked_idx = ctx.triggered_id["index"]
    # Toggle: click same card deselects
    if clicked_idx == current_idx:
        return None
    return clicked_idx


@app.callback(
    Output("pitch-graph", "figure", allow_duplicate=True),
    Output("animation-loading", "data", allow_duplicate=True),
    Output("autoplay-signal", "data", allow_duplicate=True),
    Output("play-animation-btn", "disabled", allow_duplicate=True),
    Output("play-animation-btn", "className", allow_duplicate=True),
    Input("play-animation-btn", "n_clicks"),
    State("match-selector", "value"),
    State("pass-selector", "value"),
    prevent_initial_call=True,
)
def play_animation(n_clicks, game_id, pass_idx):
    """Load tracking data and build an animated pitch figure."""
    if not n_clicks or game_id is None or pass_idx is None:
        return no_update, False, no_update, False, "play-animation-btn"

    # Load pass event and match info
    passes = _load_passes(game_id)
    pass_event = passes[int(pass_idx)]
    match_info = _get_match_info(game_id)

    # Extract animation frames from tracking data (streaming, cached)
    frames = get_animation_frames_cached(
        game_id=game_id,
        pass_event=pass_event,
        period_start_times=match_info.period_start_times,
    )

    if frames is None or len(frames) == 0:
        return no_update, False, no_update, False, "play-animation-btn"

    # Build animation figure
    home = match_info.home_team.short_name
    away = match_info.away_team.short_name
    tick = "\u2713" if pass_event.is_complete else "\u2717"
    title = (
        f"{home} vs {away} \u2014 "
        f"{pass_event.passer_name} \u2192 "
        f"{pass_event.target_name} ({tick})"
    )
    fig = build_animation_figure_3d(
        frames,
        pitch_length=pass_event.snapshot.pitch_length,
        pitch_width=pass_event.snapshot.pitch_width,
        title=title,
    )
    fig.update_layout(height=None, autosize=True)

    import time as _t
    return fig, False, _t.time(), False, "play-animation-btn"


# ---------------------------------------------------------------------------
# Panel Builder
# ---------------------------------------------------------------------------
def _build_analysis_panel(
    result: AnalysisResult,
    pass_event: PassEvent,
    match_info,
    selected_idx: int | None = None,
) -> list:
    """Build the right-side analysis panel content."""
    ranked = result.ranked_options
    children = []

    # ---- Match Info Card ----
    children.append(
        html.Div(
            className="match-info-card",
            children=[
                html.Div(
                    className="teams",
                    children=[
                        html.Span(
                            match_info.home_team.name,
                            className="text-home",
                        ),
                        html.Span(" vs ", className="vs"),
                        html.Span(
                            match_info.away_team.name,
                            className="text-away",
                        ),
                    ],
                ),
                html.Div(
                    f"{match_info.stadium_name}",
                    className="detail",
                ),
            ],
        )
    )

    # ---- Pass Event Info ----
    clock_min = pass_event.game_clock // 60
    clock_sec = pass_event.game_clock % 60
    outcome_text = "Complete" if pass_event.is_complete else "Incomplete"
    outcome_color = "#2ecc71" if pass_event.is_complete else "#e74c3c"

    children.append(
        html.Div(
            className="pass-event-info",
            children=[
                html.Div([
                    html.Span(
                        f"{pass_event.passer_name}",
                        className="passer-name",
                    ),
                    html.Span(
                        f" \u2192 {pass_event.target_name}",
                        style={"color": "#9999b3", "fontSize": "0.85rem"},
                    ),
                ]),
                html.Div([
                    html.Span(
                        f"P{pass_event.period} {clock_min:02d}:{clock_sec:02d}",
                        className="meta-tag",
                    ),
                    html.Span(
                        outcome_text,
                        className="meta-tag",
                        style={
                            "color": outcome_color,
                            "marginLeft": "6px",
                        },
                    ),
                    *(
                        [html.Span(
                            "Better Option",
                            className="meta-tag",
                            style={
                                "color": "#FFD700",
                                "marginLeft": "6px",
                            },
                        )]
                        if pass_event.better_option_type
                        else []
                    ),
                ]),
            ],
        )
    )

    # ---- Section Title ----
    children.append(
        html.Div(
            className="panel-section-title",
            children=[
                "\ud83d\udcca",
                f"PASS OPTIONS ({len(ranked)} evaluated)",
            ],
        )
    )

    # ---- Option Cards ----
    for i, opt in enumerate(ranked):
        is_actual = opt.target.player_id == pass_event.target_id
        is_selected = selected_idx is not None and i == selected_idx

        card_class = "option-card"
        if is_selected:
            card_class += " selected"
        if is_actual:
            card_class += " actual-target"

        # Rank badge
        rank_class = f"rank-{i + 1}" if i < 3 else "rank-other"

        # Player label
        num = f"#{opt.target.jersey_num}" if opt.target.jersey_num else ""
        name = opt.target.name or f"ID:{opt.target.player_id}"

        # Score color
        sc = _score_color(opt.pass_score)

        # Dimension bars
        dimensions = _build_dimension_bars(opt)

        card = html.Div(
            id={"type": "option-card", "index": i},
            className=card_class,
            n_clicks=0,
            children=[
                # Header row
                html.Div(
                    className="option-header",
                    children=[
                        html.Div(
                            className="d-flex align-items-center",
                            children=[
                                html.Div(
                                    str(i + 1),
                                    className=f"rank-badge {rank_class}",
                                ),
                                html.Span(
                                    f"{num} {name}",
                                    className="option-player",
                                ),
                                *(
                                    [html.Span("ACTUAL", className="actual-tag")]
                                    if is_actual
                                    else []
                                ),
                            ],
                        ),
                        html.Span(
                            f"{opt.pass_score:.1f}",
                            className="option-score",
                            style={"color": sc},
                        ),
                    ],
                ),
                # Dimension bars
                *dimensions,
            ],
        )
        children.append(card)

    return children


def _build_dimension_bars(opt) -> list:
    """Build the 5 dimension mini-bars for a pass option."""
    cp = opt.completion_probability
    rp = opt.receiving_pressure
    sa = opt.space_available
    ps = opt.penetration_score
    dims = [
        ("Completion", cp, 1.0, f"{cp:.0%}", "#3498db"),
        ("Zone Value", opt.zone_value, 0.60,
         f"{opt.zone_value:.3f}", "#9b59b6"),
        ("Pressure", 1 - rp / 10, 1.0,
         f"{rp:.1f}/10", "#e67e22"),
        ("Space", min(sa / 20, 1.0), 1.0,
         f"{sa:.1f}m", "#1abc9c"),
        ("Penetration", max(0, ps), 1.0,
         f"{ps:.2f}", "#2ecc71"),
    ]

    rows = []
    for label, fill_ratio, max_val, display, color in dims:
        pct = max(0, min(100, fill_ratio * 100))
        rows.append(
            html.Div(
                className="dimension-row",
                children=[
                    html.Span(label, className="dimension-label"),
                    html.Div(
                        className="dimension-bar-bg",
                        children=[
                            html.Div(
                                className="dimension-bar-fill",
                                style={
                                    "width": f"{pct:.0f}%",
                                    "backgroundColor": color,
                                },
                            ),
                        ],
                    ),
                    html.Span(display, className="dimension-value"),
                ],
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
def main():
    """Launch the Threader Dash app."""
    print("\ud83e\uddf5 Threader \u2014 Starting at http://127.0.0.1:8050")
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
