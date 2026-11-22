"""B2 v2: the three runes of the Azir / Diana lists. A rune's mechanics (exhaust: add 1 Energy; recycle:
add 1 Power of its Domain) are expanded by extract.build_class from the card type, as for the v1 runes
(graphs_code._rune)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graphs_code as GC  # noqa: E402

BUILDERS = {
    ("order_rune", "ogn-214-298"): GC._rune,
    ("chaos_rune", "ogn-166-298"): GC._rune,
    ("mind_rune", "ogn-089-298"): GC._rune,
}
