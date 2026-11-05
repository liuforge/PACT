"""Paper-wide palette (also used by the TikZ figures, figures/tikz/palette.tex, and the table style)."""
C1, C2, C3, C4 = "#421454", "#802C62", "#FF5B42", "#FFD749"  # deep purple, plum, coral, yellow
SERIES = [C1, C3, C2, C4]  # categorical order: the two dark/strong hues first, yellow (fills only) last
INK, MUTED, GRID, BAND = "#1F1A24", "#5F5866", "#E6E3EA", "#F4F3F5"


def edge(c):
    """Marker edge: yellow needs a dark outline on white, the other hues a white ring."""
    return C2 if c == C4 else "white"
