"""Damerau-Levenshtein-Distanz."""

from __future__ import annotations


def damerau_levenshtein(a: str, b: str) -> int:
    """Berechne die Damerau-Levenshtein-Distanz zwischen ``a`` und ``b``."""

    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    inf = la + lb
    da: dict[str, int] = {}
    d = [[0] * (lb + 2) for _ in range(la + 2)]
    d[0][0] = inf
    for i in range(la + 1):
        d[i + 1][0] = inf
        d[i + 1][1] = i
    for j in range(lb + 1):
        d[0][j + 1] = inf
        d[1][j + 1] = j
    for i in range(1, la + 1):
        db = 0
        for j in range(1, lb + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(
                d[i][j] + cost,  # Substitution
                d[i + 1][j] + 1,  # Insertion
                d[i][j + 1] + 1,  # Deletion
                d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),  # Transposition
            )
        da[a[i - 1]] = i
    return d[la + 1][lb + 1]


__all__ = ["damerau_levenshtein"]
