"""One key per circuit for matching event names across sources.

`driver_rounds.csv` carries FastF1's event name ("São Paulo Grand Prix"),
`circuits.csv` was typed by hand without accents ("Sao Paulo Grand Prix"), and
some events were renamed while the track stayed the same. Exact string
matching silently dropped the hand rating for such a round and split one
track's history in two, so every lookup goes through `circuit_key`.
"""

import unicodedata

# Same track, new event name. Keys and values are accent-free.
SAME_TRACK_RENAMES = {
    "Brazilian Grand Prix": "Sao Paulo Grand Prix",
}


def circuit_key(name):
    """Accent-free, whitespace-collapsed event name, with same-track renames applied."""
    if name is None or (isinstance(name, float) and name != name):
        return None
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    plain = " ".join(plain.split())
    return SAME_TRACK_RENAMES.get(plain, plain)
