"""Zone value — xT (Expected Threat) grid lookup.

Maps a pitch position to a threat value indicating how dangerous
that location is for creating goals.

The xT grid uses pitch coordinates (0,0 at bottom-left), so PFF
center-origin coordinates must be converted first.

Grid: 12 columns (x-axis, defense→attack) × 8 rows (y-axis, bottom→top)
"""

from __future__ import annotations

import numpy as np

# Simplified xT grid (12 cols × 8 rows)
# Derived from Karun Singh's Expected Threat model.
# Columns: x-axis (0→105m, left=defense, right=attack)
# Rows: y-axis (0→68m, bottom→top)
_XT_GRID = np.array([
    # col 0    col 1    col 2    col 3    col 4    col 5
    # col 6    col 7    col 8    col 9    col 10   col 11
    # ---- defensive third ----  -- midfield --  -- attacking third --

    # Row 0 (bottom touchline)
    [0.000, 0.001, 0.002, 0.005, 0.010, 0.015,
     0.020, 0.035, 0.060, 0.100, 0.150, 0.100],

    # Row 1
    [0.001, 0.002, 0.004, 0.008, 0.015, 0.025,
     0.035, 0.055, 0.090, 0.160, 0.250, 0.200],

    # Row 2
    [0.001, 0.003, 0.005, 0.010, 0.020, 0.035,
     0.050, 0.075, 0.120, 0.220, 0.350, 0.320],

    # Row 3 (center-bottom)
    [0.002, 0.003, 0.006, 0.012, 0.025, 0.040,
     0.060, 0.090, 0.150, 0.280, 0.430, 0.450],

    # Row 4 (center-top)
    [0.002, 0.003, 0.006, 0.012, 0.025, 0.040,
     0.060, 0.090, 0.150, 0.280, 0.430, 0.450],

    # Row 5
    [0.001, 0.003, 0.005, 0.010, 0.020, 0.035,
     0.050, 0.075, 0.120, 0.220, 0.350, 0.320],

    # Row 6
    [0.001, 0.002, 0.004, 0.008, 0.015, 0.025,
     0.035, 0.055, 0.090, 0.160, 0.250, 0.200],

    # Row 7 (top touchline)
    [0.000, 0.001, 0.002, 0.005, 0.010, 0.015,
     0.020, 0.035, 0.060, 0.100, 0.150, 0.100],
])


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def zone_value(
    x: float,
    y: float,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> float:
    """Look up the xT value for a PFF center-origin coordinate.

    Converts from PFF coordinates (center-origin) to pitch coordinates
    (0,0 at bottom-left) before looking up the grid.

    Returns a value in [0, ~0.45].
    """
    # PFF center-origin → pitch coordinates (0,0 at bottom-left)
    pitch_x = x + pitch_length / 2.0
    pitch_y = y + pitch_width / 2.0

    # Grid index lookup
    col = _clamp(int(pitch_x / pitch_length * 12), 0, 11)
    row = _clamp(int(pitch_y / pitch_width * 8), 0, 7)

    return float(_XT_GRID[row, col])
