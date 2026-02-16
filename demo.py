"""Threader Demo — Analyze a real pass from FIFA World Cup 2022.

Usage:
    uv run python demo.py
"""

from threader.analysis.analyzer import analyze_pass_event
from threader.data.events import extract_pass_events
from threader.data.metadata import load_match_info
from threader.viz.passes import visualize_analysis


def main():
    game_id = "3812"  # Senegal vs Netherlands
    data_root = "data/FIFA_World_Cup_2022"

    # Load match info
    match = load_match_info(f"{data_root}/Metadata/{game_id}.json")
    print(f"Match: {match.home_team.name} vs {match.away_team.name}")
    print(f"Stadium: {match.stadium_name}\n")

    # Load all pass events
    passes = extract_pass_events(
        f"{data_root}/Event Data/{game_id}.json",
        f"{data_root}/Metadata/{game_id}.json",
    )
    print(f"Total passes: {len(passes)}")
    print(f"Complete: {sum(1 for p in passes if p.is_complete)}")
    print(f"With better option annotation: {sum(1 for p in passes if p.better_option_type)}\n")

    # Pick an interesting pass (first open-play pass after kickoff)
    pass_event = passes[5]
    print(f"Analyzing pass: {pass_event.passer_name} -> {pass_event.target_name}")
    print(f"Outcome: {'Complete' if pass_event.is_complete else 'Incomplete'}")
    print(f"Game clock: {pass_event.game_clock}s (Period {pass_event.period})\n")

    # Run analysis
    result = analyze_pass_event(pass_event)

    # Print rankings
    print("=" * 60)
    print("PASS OPTIONS (ranked by Pass Score)")
    print("=" * 60)
    for i, opt in enumerate(result.ranked_options):
        marker = " <-- actual target" if opt.target.player_id == pass_event.target_id else ""
        name = f"#{opt.target.jersey_num}" if opt.target.jersey_num else f"ID:{opt.target.player_id}"
        print(f"\n  {i+1}. {name} — Pass Score: {opt.pass_score:.1f}{marker}")
        print(f"     Completion: {opt.completion_probability:.0%}")
        print(f"     Zone Value: {opt.zone_value:.3f}")
        print(f"     Pressure:   {opt.receiving_pressure:.1f}/10")
        print(f"     Space:      {opt.space_available:.1f}m")
        print(f"     Penetration:{opt.penetration_score:.2f}")

    # Visualize
    title = (
        f"{match.home_team.short_name} vs {match.away_team.short_name} — "
        f"{pass_event.passer_name} (clock: {pass_event.game_clock}s)"
    )
    fig, ax = visualize_analysis(result, top_n=3, show_all=True, title=title)
    fig.savefig("demo_output.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n\nVisualization saved to demo_output.png")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
